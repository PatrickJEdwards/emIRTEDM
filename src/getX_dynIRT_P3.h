// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#ifndef GETX_DYNIRT_P3_HPP
#define GETX_DYNIRT_P3_HPP

#include <RcppArmadillo.h>

void getX_dynIRT_P3(arma::mat &Ex,
                arma::mat &Vx,
                const arma::mat &Ebb,
                const arma::mat &omega2,
                const arma::mat &Eb,
                const arma::mat &Eystar,
                const arma::mat &Eba,
                const arma::mat &startlegis,
                const arma::mat &endlegis,
                const arma::mat &prevlegis,
                const arma::mat &xmu0,
                const arma::mat &xsigma0,
                const arma::mat &xsign,      // Nx1: legislator sign constraints (left-wing -> non-positive, right-wing -> non-negative, unconstrained)
                const int T,
                const int N,
                const arma::mat &end_session,
                const arma::mat &Ep
                );

#endif
