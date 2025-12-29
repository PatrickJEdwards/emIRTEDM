// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#ifndef GETP_DYNIRT_RW_H
#define GETP_DYNIRT_RW_H

#include <RcppArmadillo.h>

// Updates propensity means Ep (N x T) and variances Vp (N x T) in place.
// Uses residuals r_ijt = E[y*_ijt] - E[alpha_jt] - E[beta_jt] * E[x_it],
// a N(pmu, psigma) prior, and enforces sum_i p_it = 0 within each time t.
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
                    const unsigned int J);

#endif