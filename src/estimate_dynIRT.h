// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#ifndef ESTIMATE_DYNIRT_H
#define ESTIMATE_DYNIRT_H

#include <RcppArmadillo.h>

Rcpp::List estimate_dynIRT(arma::mat m_start,
                    arma::mat s_start,
                    arma::mat x_start,
                    arma::mat p_start,     // matrix of propensity starting values
                    arma::mat y,
                    arma::mat startlegis,
                    arma::mat endlegis,
                    arma::mat bill_session,
                    unsigned int T,
                    arma::mat sponsor_index,
                    arma::mat xmu0,
                    arma::mat xsigma0,
                    arma::mat item_sigma,
                    arma::mat omega2,
                    double pmu = 0.0,             // propensity p_{it} prior mean value
                    double psigma = 1.0,          // propensity p_{it} prior variance value
                    unsigned int threads = 1,
                    bool verbose = true,
                    unsigned int maxit = 2500,
                    double thresh = 1e-9,
                    unsigned int checkfreq = 50
                    ) ;

#endif
