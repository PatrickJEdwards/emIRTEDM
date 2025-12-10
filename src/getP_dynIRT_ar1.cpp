// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
#include "getP_dynIRT_ar1.h"
#include <cmath>
using namespace Rcpp; using arma::mat; using arma::uword;

static inline double sq(double x){ return x*x; }

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
                     const mat &psig20,
                     const unsigned int T,
                     const unsigned int N,
                     const unsigned int J)
{
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
  mat C(N,T,arma::fill::zeros);    // filtered var
  mat c(N,T,arma::fill::zeros);    // filtered mean
  mat Jt(N,T,arma::fill::zeros);   // smoother gain
  
  // Utilities for truncation
  auto trunc_moments = [](double mu, double var){
    const double sd = std::sqrt(std::max(var, 1e-14));
    const double a  = (0.0 - mu)/sd;
    const double log_phi = R::dnorm(a, 0.0, 1.0, /*log*/true);
    const double log_Phi = R::pnorm(a, 0.0, 1.0, /*lower*/true, /*log*/true);
    double lambda;
    if (std::isfinite(log_Phi)) lambda = std::exp(log_phi - log_Phi);
    else                        lambda = -a; // safe tail approx
    double m  = mu - sd*lambda;
    double v  = var * (1.0 - a*lambda - lambda*lambda);
    if (m > 0.0) m = 0.0;
    if (!(v > 0.0) || !std::isfinite(v)) v = 1e-12;
    return std::pair<double,double>(m,v);
  };
  
  // main loop over legislators
#pragma omp parallel for
  for (int ii = 0; ii < (int)N; ++ii) {
    // skip entirely if never serves
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
      double betaDD = std::sqrt(std::max(B2(t,0), 1e-16));
      double Gt     = (betaDD > 0.0) ? (B1(t,0)/betaDD) : 0.0;
      
      // build collapsed measurement ypp¨ = [ Σ E[y*]E[β] − Ex*Σ Eβ^2 − Σ Eβα ] / betaDD
      double EyEb = 0.0;
      for (uword j = 0; j < J; ++j) if ((int)bill_session(j,0) == t) EyEb += Eystar(ii,j) * beta(j,0);
      double ypp = (betaDD > 0.0) ? ( (EyEb - Ex(ii,t)*B2(t,0) - BA(t,0)) / betaDD ) : 0.0;
      
      double Q   = sig2_p(ii,0);
      double c_pred = pmu0(ii,0);           // mean 0 prior at entry (you can shift if you like)
      double C_pred = psig20(ii,0);
      
      double S    = Gt*Gt*C_pred + 1.0;
      double K    = (S>0.0) ? (Gt*C_pred / S) : 0.0;
      double c_f  = c_pred + K * (ypp - Gt*c_pred);
      double C_f  = (1.0 - K*Gt) * C_pred;
      
      c(ii,t)  = c_f;
      C(ii,t)  = C_f;
      Ot(ii,t) = C_pred;     // store prediction var at entry
      St(ii,t) = S;
      Kt(ii,t) = K;
    }
    
    // remaining served periods
    for (int t = t0+1; t <= t1; ++t) {
      double betaDD = std::sqrt(std::max(B2(t,0), 1e-16));
      double Gt     = (betaDD > 0.0) ? (B1(t,0)/betaDD) : 0.0;
      
      double EyEb = 0.0;
      for (uword j = 0; j < J; ++j) if ((int)bill_session(j,0) == t) EyEb += Eystar(ii,j) * beta(j,0);
      double ypp = (betaDD > 0.0) ? ( (EyEb - Ex(ii,t)*B2(t,0) - BA(t,0)) / betaDD ) : 0.0;
      
      double Q     = sig2_p(ii,0);
      double c_pred= rho_p * c(ii,t-1);
      double C_pred= rho_p*rho_p*C(ii,t-1) + Q;
      
      double S     = Gt*Gt*C_pred + 1.0;
      double K     = (S>0.0) ? (Gt*C_pred / S) : 0.0;
      double c_f   = c_pred + K * (ypp - Gt*c_pred);
      double C_f   = (1.0 - K*Gt) * C_pred;
      
      c(ii,t)  = c_f;
      C(ii,t)  = C_f;
      Ot(ii,t) = C_pred;
      St(ii,t) = S;
      Kt(ii,t) = K;
    }
    
    // ----- backward smoother (RTS) -----
    // initialize at last served
    Ep(ii,t1) = c(ii,t1);
    Vp(ii,t1) = C(ii,t1);
    
    for (int t = t1-1; t >= t0; --t) {
      double Q     = sig2_p(ii,0);
      double C_pred= rho_p*rho_p*C(ii,t) + Q;
      double Jgain = (C_pred>0.0) ? (rho_p * C(ii,t) / C_pred) : 0.0;
      
      double mean_next = Ep(ii,t+1);
      double var_next  = Vp(ii,t+1);
      
      double sm_mean = c(ii,t) + Jgain * (mean_next - rho_p * c(ii,t));
      double sm_var  = C(ii,t) + Jgain*Jgain * (var_next - C_pred);
      
      Ep(ii,t) = sm_mean;
      Vp(ii,t) = sm_var;
      Jt(ii,t) = Jgain;
    }
    
    // ----- truncate each marginal to (-inf,0] -----
    for (int t = t0; t <= t1; ++t) {
      auto mv = trunc_moments(Ep(ii,t), Vp(ii,t));
      Ep(ii,t) = mv.first;
      Vp(ii,t) = mv.second;
    }
    
    // outside service: zero
    for (int t = 0; t < t0; ++t) { Ep(ii,t)=0.0; Vp(ii,t)=0.0; }
    for (int t = t1+1; t < (int)T; ++t) { Ep(ii,t)=0.0; Vp(ii,t)=0.0; }
  } // i
}