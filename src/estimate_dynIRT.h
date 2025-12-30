// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#ifndef ESTIMATE_DYNIRT_H
#define ESTIMATE_DYNIRT_H

#include <RcppArmadillo.h>

Rcpp::List estimate_dynIRT(arma::mat m_start,
                    arma::mat s_start,
                    arma::mat x_start,
                    arma::mat p_start,         // matrix of propensity starting values
                    arma::mat y,
                    arma::mat startlegis,
                    arma::mat endlegis,
                    arma::mat prevlegis,       // N x 1 column matrix of previous contiguous legislators (== 0 when no prior contiguous legislators).
                    arma::mat bill_session,
                    unsigned int T,
                    arma::mat sponsor_index,
                    arma::mat anchor_group,    // J x 1 (0 = singleton; >0 = tied)
                    arma::mat beta_sign,       // J x 1 (-1 = beta <= 0, 1 = beta >= 0, 0 = beta unconstrained)
                    arma::mat xmu0,
                    arma::mat xsigma0,
                    arma::mat xsign,           // Nx1: legislator sign constraints (left-wing -> non-positive, right-wing -> non-negative, unconstrained)
                    arma::mat item_sigma,
                    arma::mat omega2,
                    double rho_p,              // AR(1) coefficient in (-1,1). Use 1.0 for random walk.
                    arma::mat sig2_p,          // N x 1 innovation variance for propensity per legislator
                    arma::mat pmu0,            // N x 1 prior mean for p at first served period (often 0 and large).
                    arma::mat psigma0,         // N x 1 prior var for p at first served period (often 0 and large).
                    unsigned int threads = 1,
                    bool verbose = true,
                    unsigned int maxit = 2500,
                    double thresh = 1e-9,
                    unsigned int checkfreq = 50
                    ) ;

#endif
