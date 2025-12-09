// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-
#ifndef GETMS_GN_DYNIRT_H
#define GETMS_GN_DYNIRT_H

#include <RcppArmadillo.h>

// Gauss–Newton/Laplace item update on (m_jt, s_jt) for each item j.
// Inputs:
//   Eystar      : N x J, E[y*_{ijt}]
//   Ex          : N x T, E[x_{it}]
//   Ep          : N x T, E[p_{it}]
//   bill_session: J x 1, integer t(j) in {0,...,T-1}
//   ones_col    : N x T, 1 for in-service (i,t), 0 otherwise
//   mu_ms       : 2 x J prior means for (m,s) in item order (or 2 x 1 to broadcast)
//   Sigma_ms    : 2 x 2 prior covariance for (m,s) (shared across items)
//   max_newton  : small fixed number of GN steps per item (e.g., 3–5)
//
// Outputs (filled in place, length-J unless noted):
//   Em          : J x 1  (mode / posterior mean of m)
//   Es          : J x 1  (mode / posterior mean of s)
//   Vm2         : 2 x 2 x J Laplace covariance at the mode
//   Ea          : J x 1  (alpha_hat = s^2 - m^2 at the mode)
//   Eb          : J x 1  (beta_hat  = 2(m - s) at the mode)
//   Ebb         : J x 1  ≈ E[β^2] via delta method = β_hat^2 + Var(β)
//   Eba         : J x 1  ≈ E[β α] via delta method = β_hat α_hat + Cov(β,α)
void getMS_GN_dynIRT(arma::mat &Em, arma::mat &Es, arma::cube &Vm2,
                     arma::mat &Ea, arma::mat &Eb,
                     arma::mat &Ebb, arma::mat &Eba,
                     const arma::mat &Eystar,
                     const arma::mat &Ex,
                     const arma::mat &Ep,
                     const arma::mat &bill_session,
                     const arma::mat &ones_col,
                     const arma::mat &mu_ms,          // 2 x J (or 2 x 1 broadcast)
                     const arma::mat &Sigma_ms,       // 2 x 2
                     const unsigned int max_newton = 4);

#endif
