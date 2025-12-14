// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
#include "getP_dynIRT_ar1.h"
#include <cmath>
using namespace Rcpp; using arma::mat; using arma::uword;

static inline double sq(double x){ return x*x; }

// logdiffexp helper: returns log(exp(b) - exp(a)) assuming b >= a
static inline double log_diff_exp(double log_b, double log_a){
  if (log_b < log_a) std::swap(log_b, log_a);
  double x = std::exp(log_a - log_b);
  if (x >= 1.0) return -INFINITY;
  return log_b + std::log1p(-x);
}

// Truncated-Normal moments on a finite box [L, U]
static inline std::pair<double,double>
trunc_box(double mu, double var, double L, double U){
  const double VAR_FLOOR = 1e-12;
  var = std::max(var, VAR_FLOOR);
  if (!(U > L)) return { std::min(U,std::max(L,mu)), VAR_FLOOR };

  const double sd = std::sqrt(var);
  const double a  = (L - mu)/sd;
  const double b  = (U - mu)/sd;

  const double log_phi_a = R::dnorm(a, 0.0, 1.0, /*log*/true);
  const double log_phi_b = R::dnorm(b, 0.0, 1.0, /*log*/true);
  const double log_Phi_a = R::pnorm(a, 0.0, 1.0, /*lower*/true,  /*log*/true);
  const double log_Phi_b = R::pnorm(b, 0.0, 1.0, /*lower*/true,  /*log*/true);

  double logZ = log_diff_exp(log_Phi_b, log_Phi_a);
  if (!std::isfinite(logZ)) return { std::min(U,std::max(L,mu)), VAR_FLOOR };

  const double r_a = std::exp(log_phi_a - logZ);
  const double r_b = std::exp(log_phi_b - logZ);

  const double mean = mu + sd * (r_a - r_b);
  const double term = (a * r_a - b * r_b);
  double variance   = var * (1.0 + term - (r_a - r_b)*(r_a - r_b));
  if (!(variance > 0.0) || !std::isfinite(variance)) variance = VAR_FLOOR;

  double mean_box = std::min(U, std::max(L, mean));
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
  const double EPS   = 1e-8;
  
  // --------- precompute per-time item sums ----------
  mat B2(T,1,arma::fill::zeros);   // sum beta^2
  mat B1(T,1,arma::fill::zeros);   // sum beta
  mat BA(T,1,arma::fill::zeros);   // sum beta*alpha
  
  // We assume items are grouped by session; if not, we still loop safely.
  for (uword j = 0; j < J; ++j) {
    int t = static_cast<int>(bill_session(j,0));
    if (t < 0 || t >= (int)T) continue;
    const double bj = beta(j,0);
    const double aj = alpha(j,0);
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
      for (int tt=0; tt<T; ++tt){ Ep(ii,tt)=0.0; Vp(ii,tt)=0.0; }
      continue;
    }
    
    // ----- forward filter -----
    // prior at entry
    {
      int t = t0;
      const double betaDD = std::sqrt(std::max(B2(t,0), EPS));
      const double Gt     = (betaDD > 0.0) ? (   B1(t,0) / betaDD ) : 0.0;  // OLD, apparently missing neg. sign
      //const double Gt     = (betaDD > 0.0) ? ( - B1(t,0) / betaDD ) : 0.0;
      
      
      // collapsed measurement for p:
      // ypp = [ Σ E[y*]E[β] − Ex*Σ Eβ^2 − Σ Eβα ] / betaDD
      double EyEb = 0.0;
      for (uword j = 0; j < J; ++j)
        if ((int)bill_session(j,0) == t) EyEb += Eystar(ii,j) * beta(j,0);
      
      const double ypp = (betaDD > 0.0) ? ( (EyEb - Ex(ii,t)*B2(t,0) - BA(t,0)) / betaDD ) : 0.0;

      const double Q      = sig2_p(ii,0);
      const double var_st = Q / std::max(1e-8, (1.0 - rho_p*rho_p));
      const double C_pred = std::max(psigma0(ii,0), var_st);
      const double c_pred = pmu0(ii,0);

      St(ii,t)    = Gt*Gt*C_pred + 1.0;
      Kt(ii,t)    = (St(ii,t)>EPS)? (Gt*C_pred / St(ii,t)) : 0.0;
      C(ii,t)     = std::max((1.0 - Kt(ii,t)*Gt) * C_pred, 1e-12);
      c(ii,t)     = c_pred + Kt(ii,t) * (ypp - Gt*c_pred);
      
      // **soft truncation** of filtered posterior
      auto mv = trunc_box(c(ii,t), C(ii,t), P_MIN, P_MAX);
      c(ii,t) = mv.first;
      C(ii,t) = mv.second;
      
      Ot(ii,t)   = C_pred;
    }
    
    // remaining served periods
    for (int t = t0+1; t <= t1; ++t) {
      
      const double betaDD = std::sqrt(std::max(B2(t,0), EPS));
      //const double Gt     = (betaDD > 0.0) ? ( - B1(t,0) / betaDD ) : 0.0;
      const double Gt     = (betaDD > 0.0) ? (   B1(t,0) / betaDD ) : 0.0; // OLD, apparently missing neg. sign

      double EyEb = 0.0;
      for (uword j = 0; j < J; ++j)
        if ((int)bill_session(j,0) == t) EyEb += Eystar(ii,j) * beta(j,0);
        
      const double ypp = (betaDD > 0.0) ? ( (EyEb - Ex(ii,t)*B2(t,0) - BA(t,0)) / betaDD ) : 0.0;

      const double Q   = sig2_p(ii,0);
      const double C_pred = rho_p*rho_p*C(ii,t-1) + Q;
      const double c_pred = rho_p * c(ii,t-1);

      St(ii,t)    = Gt*Gt*C_pred + 1.0;
      Kt(ii,t)    = (St(ii,t)>EPS)? (Gt*C_pred / St(ii,t)) : 0.0;
      C(ii,t)     = std::max((1.0 - Kt(ii,t)*Gt) * C_pred, 1e-12);
      c(ii,t)     = c_pred + Kt(ii,t) * (ypp - Gt*c_pred);
      
      auto mv     = trunc_box(c(ii,t), C(ii,t), P_MIN, P_MAX);
      c(ii,t)     = mv.first;
      C(ii,t)     = mv.second;
      
      Ot(ii,t)    = C_pred;
    }
    
    // ----- backward smoother (RTS) -----
    // initialize at last served
    Ep(ii,t1) = c(ii,t1);
    Vp(ii,t1) = C(ii,t1);
    {
      auto mv = trunc_box(Ep(ii,t1), Vp(ii,t1), P_MIN, P_MAX);
      Ep(ii,t1) = mv.first;
      Vp(ii,t1) = mv.second;
    }

    for (int t = t1-1; t >= t0; --t) {
      const double C_pred = rho_p*rho_p*C(ii,t) + sig2_p(ii,0);
      const double Jgain  = (C_pred>0.0) ? (rho_p * C(ii,t) / C_pred) : 0.0;
      
      Vp(ii,t) = std::max(C(ii,t) + Jgain*Jgain * (Vp(ii,t+1) - C_pred), 1e-12);
      Ep(ii,t) = c(ii,t) + Jgain * (Ep(ii,t+1) - rho_p * c(ii,t));
      
      auto mv  = trunc_box(Ep(ii,t), Vp(ii,t), P_MIN, P_MAX);
      Ep(ii,t) = mv.first;
      Vp(ii,t) = mv.second;
    }

    // outside service: zero
    for (int t=0; t<t0; ++t){ Ep(ii,t)=0.0; Vp(ii,t)=0.0; }
    for (int t=t1+1; t<(int)T; ++t){ Ep(ii,t)=0.0; Vp(ii,t)=0.0; }
  } // ii
}