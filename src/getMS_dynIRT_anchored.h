#pragma once
#include <RcppArmadillo.h>
using arma::mat; using arma::cube; using arma::ivec;

void getMS_dynIRT_anchored(
    mat& curEm, 
    mat& curEs, 
    mat& curVm, 
    mat& curVs, 
    mat& curCms,
    const mat& curEystar, 
    const mat& curEx, 
    const mat& curEp,
    const mat& bill_session, 
    const mat& sponsor_index,
    const mat& item_sigma, 
    const cube& curEx2x2,
    const mat& startlegis, 
    const mat& endlegis,
    unsigned int nJ, 
    unsigned int nN, 
    unsigned int T,
    const ivec& anchor_group,
    bool prior_use_first_occurrence = true,
    unsigned int newton_maxit = 10,
    double newton_tol = 1e-8,
    double ridge = 1e-8
);