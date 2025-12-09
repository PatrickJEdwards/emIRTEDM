#pragma once
#include <RcppArmadillo.h>

// Update (m,s) for all items via Laplace/second-order VB.
// curEm,curEs: IN/OUT means of m,s (J x 1)
// curVm,curVs,curCms: OUT posterior variances Var(m), Var(s), and Cov(m,s) per item (J x 1)
// curEystar: N x J   (E[y*])
// curEx:     N x T   (E[x])
// curEp:     N x T   (E[p])
// bill_session: J x 1 (integer in [0..T-1])
// sponsor_index: J x 1 (row index of sponsor in [0..N-1])
// item_sigma: 2x2 prior covariance for (m,s)
// curEx2x2:  2x2xT cube with [[S0, Sx],[Sx, Sx2]] per t (from getEx2x2_dynIRT)
// startlegis, endlegis: N x 1 service windows
// nJ, nN, T: sizes
// newton_maxit/newton_tol/ridge: optimizer controls
void getMS_dynIRT(arma::mat& curEm,
                  arma::mat& curEs,
                  arma::mat& curVm,
                  arma::mat& curVs,
                  arma::mat& curCms,
                  const arma::mat& curEystar,
                  const arma::mat& curEx,
                  const arma::mat& curEp,
                  const arma::mat& bill_session,
                  const arma::mat& sponsor_index,
                  const arma::mat& item_sigma,
                  const arma::cube& curEx2x2,
                  const arma::mat& startlegis,
                  const arma::mat& endlegis,
                  unsigned int nJ,
                  unsigned int nN,
                  unsigned int T,
                  unsigned int newton_maxit = 10,
                  double newton_tol = 1e-8,
                  double ridge = 1e-8);