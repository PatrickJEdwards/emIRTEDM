// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#ifndef GETP_DYNIRT_AR1_H
#define GETP_DYNIRT_AR1_H

#include <RcppArmadillo.h>

// Kalman smoother update for p_it with AR(1) prior and upper truncation at 0.
// Replaces the i.i.d. per-time update.
// Inputs:
//   Ep, Vp        : N x T IN/OUT (will be overwritten with smoothed truncated moments)
//   Eystar        : N x J
//   alpha, beta   : J x 1  (means at current iteration)
//   Ex            : N x T  (means of x)
//   bill_session  : J x 1  (0..T-1)
//   startlegis,endlegis : N x 1
//   rho_p         : AR(1) coefficient in (-1,1). Use 1.0 for random walk.
//   sig2_p        : N x 1 innovation variance per legislator (Q_i). Use a scalar if you prefer.
//   pmu0, psigma0  : N x 1 prior mean/var for p at first served period (often 0 and large).
//   T, N, J       : sizes
void getP_dynIRT_ar1(arma::mat &Ep,
                     arma::mat &Vp,
                     const arma::mat &Eystar,
                     const arma::mat &alpha,
                     const arma::mat &beta,
                     const arma::mat &Ex,
                     const arma::mat &bill_session,
                     const arma::mat &startlegis,
                     const arma::mat &endlegis,
                     const double      rho_p,
                     const arma::mat  &sig2_p,   // N x 1
                     const arma::mat  &pmu0,     // N x 1
                     const arma::mat  &psigma0,   // N x 1
                     const unsigned int T,
                     const unsigned int N,
                     const unsigned int J);