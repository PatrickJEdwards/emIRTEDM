// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace Rcpp;

// Upper-truncated Normal moments for N(mu, var) truncated to (-inf, 0]
// Returns {E, Var} after truncation.
static inline std::pair<double,double>
  trunc_upper0(double mu, double var) {
    const double VAR_FLOOR = 1e-12;
    var = std::max(var, VAR_FLOOR);
    
    const double sd = std::sqrt(var);
    const double a  = (0.0 - mu) / sd;
    
    const double log_phi = R::dnorm(a, 0.0, 1.0, /*log*/true);
    const double log_Phi = R::pnorm(a, 0.0, 1.0, /*lower_tail*/true, /*log_p*/true);
    
    double lambda;
    if (std::isfinite(log_Phi)) {
      lambda = std::exp(log_phi - log_Phi);   // phi/Phi
    } else {
      // very extreme tail: phi/Phi ~ -a
      lambda = -a;
    }
    
    double m = mu - sd * lambda;
    double v = var * (1.0 - a * lambda - lambda * lambda);
    
    if (m > 0.0) m = 0.0;
    if (!(v > 0.0) || !std::isfinite(v)) v = VAR_FLOOR;
    
    return {m, v};
  }

// helper: allow scalar 1x1 or N x 1 vector
static inline double get_scalar_or_vec(const arma::mat &M, unsigned int i, unsigned int N, const char* name) {
  if (M.n_rows == 1 && M.n_cols == 1) return M(0,0);
  if (M.n_rows == N && M.n_cols >= 1) return M(i,0);
  Rcpp::stop("%s must be 1x1 or N x 1. Got %d x %d.", name, (int)M.n_rows, (int)M.n_cols);
  return NA_REAL;
}

// Random-walk update for p_it (p_it <= 0): Kalman filter + RTS smoother + truncation.
// [[Rcpp::export()]]
void getP_dynIRT_rw(arma::mat &Ep,                 // N x T (updated, smoothed means)
                    arma::mat &Vp,                 // N x T (updated, smoothed vars)
                    const arma::mat &Eystar,       // N x J
                    const arma::mat &alpha,        // J x 1
                    const arma::mat &beta,         // J x 1
                    const arma::mat &x,            // N x T
                    const arma::mat &bill_session, // J x 1 (0..T-1)
                    const arma::mat &startlegis,   // N x 1
                    const arma::mat &endlegis,     // N x 1
                    const arma::mat &sig2_p,       // 1x1 or N x 1 innovation variance for RW
                    const arma::mat &pmu0,         // 1x1 or N x 1 prior mean at entry
                    const arma::mat &psigma0,      // 1x1 or N x 1 prior variance at entry
                    const unsigned int T,
                    const unsigned int N,
                    const unsigned int J) {
  
  const double EPSV = 1e-12;
  
  // basic dimension checks
  if (Eystar.n_rows != N || Eystar.n_cols != J) Rcpp::stop("Eystar must be N x J");
  if (alpha.n_rows  != J || alpha.n_cols  < 1) Rcpp::stop("alpha must be J x 1");
  if (beta.n_rows   != J || beta.n_cols   < 1) Rcpp::stop("beta must be J x 1");
  if (x.n_rows      != N || x.n_cols      != T) Rcpp::stop("x must be N x T");
  if (bill_session.n_rows != J || bill_session.n_cols < 1) Rcpp::stop("bill_session must be J x 1");
  
  // For each legislator i: build per-period residual averages, then RW filter+smoother.
  for (unsigned int i = 0; i < N; ++i) {
    
    const int t0 = (int)startlegis(i,0);
    const int t1 = (int)endlegis(i,0);
    
    // out-of-range / never serves
    if (t0 < 0 || t1 < t0 || t1 >= (int)T) {
      for (unsigned int tt = 0; tt < T; ++tt) { Ep(i,tt) = 0.0; Vp(i,tt) = 0.0; }
      continue;
    }
    
    const double q  = std::max(get_scalar_or_vec(sig2_p,  i, N, "sig2_p"),  0.0);
    const double m0 = get_scalar_or_vec(pmu0,    i, N, "pmu0");
    const double C0 = std::max(get_scalar_or_vec(psigma0, i, N, "psigma0"), EPSV);
    
    // 1) collapse item-level residuals into per-period mean residual y_t with variance R_t = 1/n_t
    std::vector<double> sum_r(T, 0.0);
    std::vector<unsigned int> n_t(T, 0);
    
    for (unsigned int j = 0; j < J; ++j) {
      const int tt = (int)bill_session(j,0);
      if (tt < t0 || tt > t1) continue; // only periods served by i
      
      const double r = Eystar(i,j) - alpha(j,0) - beta(j,0) * x(i, (unsigned)tt);
      if (!std::isfinite(r)) continue;
      
      sum_r[(unsigned)tt] += r;
      n_t[(unsigned)tt]   += 1;
    }
    
    std::vector<double> y(T, 0.0);
    std::vector<double> R(T, 1e12); // big => almost no info if n_t==0
    for (int tt = t0; tt <= t1; ++tt) {
      if (n_t[(unsigned)tt] > 0) {
        y[(unsigned)tt] = sum_r[(unsigned)tt] / (double)n_t[(unsigned)tt];
        R[(unsigned)tt] = 1.0 / (double)n_t[(unsigned)tt];
      } else {
        // no items observed for this (i,t): propagate only
        y[(unsigned)tt] = 0.0;
        R[(unsigned)tt] = 1e12;
      }
    }
    
    // 2) Kalman filter (random walk): store filtered means/vars and predicted variances for smoothing
    std::vector<double> m_f(T, 0.0), C_f(T, 0.0);
    std::vector<double> P_pred(T, 0.0);  // predicted variance for p_t before measurement
    
    for (int tt = t0; tt <= t1; ++tt) {
      double a, P;
      
      if (tt == t0) {
        a = m0;
        P = C0;
      } else {
        a = m_f[(unsigned)(tt-1)];
        P = C_f[(unsigned)(tt-1)] + q;     // RW prediction variance
      }
      
      P = std::max(P, EPSV);
      P_pred[(unsigned)tt] = P;
      
      // measurement update with y_t ~ N(p_t, R_t)
      const double S = P + R[(unsigned)tt];
      const double K = (S > EPSV) ? (P / S) : 0.0;
      
      double m = a + K * (y[(unsigned)tt] - a);
      double C = (1.0 - K) * P;
      C = std::max(C, EPSV);
      
      // enforce p <= 0 (approximate truncation)
      auto mv = trunc_upper0(m, C);
      m_f[(unsigned)tt] = mv.first;
      C_f[(unsigned)tt] = mv.second;
    }
    
    // 3) RTS smoother
    // initialize at t1
    for (unsigned int tt = 0; tt < (unsigned)T; ++tt) {
      if ((int)tt < t0 || (int)tt > t1) { Ep(i,tt) = 0.0; Vp(i,tt) = 0.0; }
    }
    
    Ep(i,(unsigned)t1) = m_f[(unsigned)t1];
    Vp(i,(unsigned)t1) = C_f[(unsigned)t1];
    
    // smooth backwards
    for (int tt = t1 - 1; tt >= t0; --tt) {
      const double Ctt = C_f[(unsigned)tt];
      const double Pn1 = std::max(C_f[(unsigned)tt] + q, EPSV); // predicted var at t+1 given filtered at t
      const double Jg  = Ctt / Pn1;
      
      double ms = m_f[(unsigned)tt] + Jg * (Ep(i,(unsigned)(tt+1)) - m_f[(unsigned)tt]);
      double Vs = Ctt + Jg*Jg * (Vp(i,(unsigned)(tt+1)) - Pn1);
      Vs = std::max(Vs, EPSV);
      
      auto mv = trunc_upper0(ms, Vs);
      Ep(i,(unsigned)tt) = mv.first;
      Vp(i,(unsigned)tt) = mv.second;
    }
  }
  
  return;
}
