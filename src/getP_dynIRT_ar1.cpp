// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
#include "getP_dynIRT_ar1.h"
#include <cmath>
using namespace Rcpp; using arma::mat; using arma::uword;

static inline double sq(double x){ return x*x; }

static inline double clamp(double v, double lo, double hi){
  return std::min(hi, std::max(lo, v));
}

// logdiffexp helper: returns log(exp(b) - exp(a)) assuming b >= a
static inline double log_diff_exp(double log_b, double log_a){
  if (log_b < log_a) std::swap(log_b, log_a);
  // log(exp(log_b) - exp(log_a)) = log_b + log(1 - exp(log_a - log_b))
  double x = std::exp(log_a - log_b);
  // guard tiny/negative due to rounding
  if (x >= 1.0) return -INFINITY;
  return log_b + std::log1p(-x);
}

// Truncated-Normal moments on a finite box [L, U]
static inline std::pair<double,double>
  trunc_box(double mu, double var, double L, double U){
    const double VAR_FLOOR = 1e-12;
    var = std::max(var, VAR_FLOOR);
    if (!(U > L)) { // invalid interval: fall back to clamp
      return { clamp(mu, L, U), VAR_FLOOR };
    }
    
    const double sd = std::sqrt(var);
    const double a = (L - mu)/sd;
    const double b = (U - mu)/sd;
    
    // log-phi, log-Phi for stability
    const double log_phi_a = R::dnorm(a, 0.0, 1.0, /*log*/true);
    const double log_phi_b = R::dnorm(b, 0.0, 1.0, /*log*/true);
    const double log_Phi_a = R::pnorm(a, 0.0, 1.0, /*lower*/true,  /*log*/true);
    const double log_Phi_b = R::pnorm(b, 0.0, 1.0, /*lower*/true,  /*log*/true);
    
    // Z = Phi(b) - Phi(a) in log-space
    double logZ = log_diff_exp(log_Phi_b, log_Phi_a);
    
    // If Z underflows, snap to nearest boundary
    if (!std::isfinite(logZ)) {
      double m = clamp(mu, L, U);
      return { m, VAR_FLOOR };
    }
    
    // r_a = phi(a)/Z, r_b = phi(b)/Z computed in log domain
    const double r_a = std::exp(log_phi_a - logZ);
    const double r_b = std::exp(log_phi_b - logZ);
    
    // E[X | L<=X<=U] and Var[X | ...] for X~N(mu,var)
    const double mean = mu + sd * (r_a - r_b);
    const double term = (a * r_a - b * r_b);
    double variance = var * (1.0 + term - (r_a - r_b) * (r_a - r_b));
    if (!(variance > 0.0) || !std::isfinite(variance)) variance = VAR_FLOOR;
    
    // Hard safety (rarely touched if math above is fine)
    double mean_box = clamp(mean, L, U);
    return { mean_box, variance };
  }


void getP_dynIRT_ar1(mat &Ep, mat &Vp,
                     const mat &Eystar,
                     const mat &alpha,
                     const mat &beta,
                     const mat &Ex,
                     const mat &bill_session,
                     const mat &startlegis,
                     const mat &endlegis,
                     const double rho_p,
                     const mat &sig2_p,
                     const mat &pmu0,
                     const mat &psigma0,
                     const unsigned int T,
                     const unsigned int N,
                     const unsigned int J)
{
  
  // Box constraints for p_it
  const double P_MIN = -10.0;   // absolute smallest (most negative)
  const double P_MAX =  0.0;    // absolute largest
  
  // --------- precompute per-time item sums ----------
  mat B2(T,1,arma::fill::zeros);   // sum beta^2
  mat B1(T,1,arma::fill::zeros);   // sum beta
  mat BA(T,1,arma::fill::zeros);   // sum beta*alpha
  
  // We assume items are grouped by session; if not, we still loop safely.
  for (uword j = 0; j < J; ++j) {
    int t = static_cast<int>(bill_session(j,0));
    if (t < 0 || t >= (int)T) continue;
    double bj = beta(j,0);
    double aj = alpha(j,0);
    B2(t,0) += bj*bj;
    B1(t,0) += bj;
    BA(t,0) += bj*aj;
  }
  
  // --------- Kalman forward/backward per legislator ----------
  // work arrays (per i,t)
  mat Ot(N,T,arma::fill::zeros);   // prediction var
  mat St(N,T,arma::fill::zeros);   // innovation var
  mat Kt(N,T,arma::fill::zeros);   // Kalman gain
  mat C (N,T,arma::fill::zeros);   // filtered var (post-truncation)
  mat c (N,T,arma::fill::zeros);   // filtered mean (post-truncation)
  mat Jt(N,T,arma::fill::zeros);   // smoother gain
  
  
  // main loop over legislators
#pragma omp parallel for
  for (int ii = 0; ii < (int)N; ++ii) {
    
    int t0 = static_cast<int>(startlegis(ii,0));
    int t1 = static_cast<int>(endlegis(ii,0));
    if (t0 < 0 || t1 < t0 || t1 >= (int)T) {
      for (uword t=0;t<T;++t){ Ep(ii,t)=0.0; Vp(ii,t)=0.0; }
      continue;
    }
    
    // ----- forward filter -----
    // prior at entry
    {
      int t = t0;
      const double betaDD = std::sqrt(std::max(B2(t,0), 1e-8));
      //const double Gt     = (betaDD > 0.0) ? (   B1(t,0) / betaDD ) : 0.0;  // OLD, apparently missing neg. sign
      const double Gt     = (betaDD > 0.0) ? ( - B1(t,0) / betaDD ) : 0.0;
      
      
      // collapsed measurement for p:
      // ypp = [ Σ E[y*]E[β] − Ex*Σ Eβ^2 − Σ Eβα ] / betaDD
      double EyEb = 0.0;
      for (uword j = 0; j < J; ++j)
        if ((int)bill_session(j,0) == t) EyEb += Eystar(ii,j) * beta(j,0);
      const double ypp = (betaDD > 0.0)
        ? ( (EyEb - Ex(ii,t)*B2(t,0) - BA(t,0)) / betaDD )
        : 0.0;

      const double Q      = sig2_p(ii,0);
      const double var_st = Q / std::max(1e-8, (1.0 - rho_p*rho_p));
      double c_pred = pmu0(ii,0);
      double C_pred = std::max(psigma0(ii,0), var_st);

      const double S    = Gt*Gt*C_pred + 1.0;
      const double K    = (S>0.0) ? (Gt*C_pred / S) : 0.0;
      double c_f        = c_pred + K * (ypp - Gt*c_pred);
      double C_f        = (1.0 - K*Gt) * C_pred;

      // safety clamp, then **truncate to [P_MIN,P_MAX]**
      c_f = clamp(c_f, P_MIN, P_MAX);
      C_f = std::max(C_f, 1e-12);
      auto mv_f = trunc_box(c_f, C_f, P_MIN, P_MAX);

      c(ii,t)  = mv_f.first;
      C(ii,t)  = mv_f.second;
      Ot(ii,t) = C_pred;
      St(ii,t) = S;
      Kt(ii,t) = K;
    }
    
    // remaining served periods
    for (int t = t0+1; t <= t1; ++t) {
      const double betaDD = std::sqrt(std::max(B2(t,0), 1e-8));
      // const double Gt     = (betaDD > 0.0) ? (   B1(t,0) / betaDD ) : 0.0; // OLD, apparently missing neg. sign
      const double Gt     = (betaDD > 0.0) ? ( - B1(t,0) / betaDD ) : 0.0;
      double EyEb = 0.0;
      for (uword j = 0; j < J; ++j)
        if ((int)bill_session(j,0) == t) EyEb += Eystar(ii,j) * beta(j,0);
      const double ypp = (betaDD > 0.0)
        ? ( (EyEb - Ex(ii,t)*B2(t,0) - BA(t,0)) / betaDD )
        : 0.0;

      const double Q   = sig2_p(ii,0);
      double c_pred    = rho_p * c(ii,t-1);
      double C_pred    = rho_p*rho_p*C(ii,t-1) + Q;

      const double S   = Gt*Gt*C_pred + 1.0;
      const double K   = (S>0.0) ? (Gt*C_pred / S) : 0.0;
      double c_f       = c_pred + K * (ypp - Gt*c_pred);
      double C_f       = (1.0 - K*Gt) * C_pred;

      // safety clamp, then **truncate to [P_MIN,P_MAX]**
      c_f = clamp(c_f, P_MIN, P_MAX);
      C_f = std::max(C_f, 1e-12);
      auto mv_f = trunc_box(c_f, C_f, P_MIN, P_MAX);

      c(ii,t)  = mv_f.first;
      C(ii,t)  = mv_f.second;
      Ot(ii,t) = C_pred;
      St(ii,t) = S;
      Kt(ii,t) = K;
    }
    
    // ----- backward smoother (RTS) -----
    // initialize at last served
    Ep(ii,t1) = c(ii,t1);
    Vp(ii,t1) = C(ii,t1);

    for (int t = t1-1; t >= t0; --t) {
      const double Q     = sig2_p(ii,0);
      const double C_pred= rho_p*rho_p*C(ii,t) + Q;
      const double Jgain = (C_pred>0.0) ? (rho_p * C(ii,t) / C_pred) : 0.0;

      const double mean_next = Ep(ii,t+1);
      const double var_next  = Vp(ii,t+1);

      double sm_mean = c(ii,t) + Jgain * (mean_next - rho_p * c(ii,t));
      double sm_var  = C(ii,t) + Jgain*Jgain * (var_next - C_pred);

      // safety clamp, then **truncate to [P_MIN,P_MAX]**
      sm_mean = clamp(sm_mean, P_MIN, P_MAX);
      sm_var  = std::max(sm_var, 1e-12);
      auto mv = trunc_box(sm_mean, sm_var, P_MIN, P_MAX);

      Ep(ii,t) = mv.first;
      Vp(ii,t) = mv.second;
      Jt(ii,t) = Jgain;
    }

    // outside service: zero
    for (int t = 0; t < t0; ++t) { Ep(ii,t)=0.0; Vp(ii,t)=0.0; }
    for (int t = t1+1; t < (int)T; ++t) { Ep(ii,t)=0.0; Vp(ii,t)=0.0; }
  } // ii
}