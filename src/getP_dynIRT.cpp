// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
#include "getP_dynIRT.h"
using namespace Rcpp;

// Updates Ep (means) and Vp (variances) for propensities p_it with p_it <= 0.
// Conjugate Normal update per (i,t), then apply upper-truncated Normal at 0.
void getP_dynIRT(arma::mat &Ep,                 // N x T (updated)
                 arma::mat &Vp,                 // N x T (updated)
                 const arma::mat &Eystar,       // N x J
                 const arma::mat &alpha,        // J x 1
                 const arma::mat &beta,         // J x 1
                 const arma::mat &x,            // N x T
                 const arma::mat &bill_session, // J x 1
                 const arma::mat &startlegis,   // N x 1
                 const arma::mat &endlegis,     // N x 1
                 const double pmu,              // prior mean
                 const double psigma,           // prior variance
                 const unsigned int T,
                 const unsigned int N,
                 const unsigned int J) {
  
  const double prior_prec = 1.0 / psigma;
  const double EPS_VAR = 1e-12;
  
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int t = 0; t < T; ++t) {
      
      // Out-of-service cells
      if (t < (unsigned) startlegis(i,0) || t > (unsigned) endlegis(i,0)) {
        Ep(i,t) = 0.0;
        Vp(i,t) = 0.0;
        continue;
      }
      
      // Sum residuals r_ijt = E[y*_{ijt}] - E[α_{jt}] - E[β_{jt}] E[x_{it}] for items in session t
      double sum_r = 0.0;
      unsigned int n_it = 0;
      for (unsigned int j = 0; j < J; ++j) {
        if (bill_session(j,0) == static_cast<double>(t)) {
          const double r_ijt = Eystar(i,j) - alpha(j,0) - beta(j,0) * x(i,t);
          sum_r += r_ijt;
          ++n_it;
        }
      }
      
      // Unconstrained Normal posterior for p_it
      const double post_prec = prior_prec + static_cast<double>(n_it);
      const double post_var  = 1.0 / post_prec;
      const double post_sd   = std::sqrt(post_var);
      const double post_mean = post_var * (prior_prec * pmu + sum_r);
      
      // Upper-truncate at 0: a = (0 - μ) / σ
      const double a = (0.0 - post_mean) / post_sd;
      
      // Use R math: log φ(a), log Φ(a)
      const double log_phi = R::dnorm(a, 0.0, 1.0, /*give_log=*/true);
      const double log_Phi = R::pnorm(a, 0.0, 1.0, /*lower_tail=*/true, /*log_p=*/true);
      
      // λ(a) = φ(a)/Φ(a) computed in log-space; fall back if Φ underflows
      double lambda;
      if (std::isfinite(log_Phi)) {
        lambda = std::exp(log_phi - log_Phi);
      } else {
        // Extreme tail safeguard: λ(a) ~ -a for very small Φ(a)
        lambda = -a;
      }
      
      // Truncated mean/variance for (-∞, 0]
      double mean_trunc = post_mean - post_sd * lambda;
      double var_trunc  = post_var * (1.0 - a * lambda - lambda * lambda);
      
      // Enforce bound and positivity
      if (mean_trunc > 0.0) mean_trunc = 0.0;
      if (!(var_trunc > 0.0) || !std::isfinite(var_trunc)) var_trunc = EPS_VAR;
      
      Ep(i,t) = mean_trunc;
      Vp(i,t) = var_trunc;
    }
  }
  return;
}
