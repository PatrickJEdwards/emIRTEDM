#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "estimate_dynIRT.h"

RcppExport SEXP dynIRT_estimate(SEXP m_startSEXP,        // item motion 'm' start values
                                 SEXP s_startSEXP,       // item status quo 's' start values
                                 SEXP x_startSEXP,
                                 SEXP p_startSEXP,       // propensity p_{it} start values
                                 SEXP ySEXP,
                                 SEXP startlegisSEXP,
                                 SEXP endlegisSEXP,
                                 SEXP prevlegisSEXP,     // N x 1 column matrix of previous contiguous legislators (== 0 when no prior contiguous legislators).
                                 SEXP bill_sessionSEXP,
                                 SEXP TSEXP,
                                 SEXP sponsor_indexSEXP, // length J, 0-based row index of the sponsor MP for item j
                                 SEXP anchor_groupSEXP,  // J x 1 (0 = singleton; >0 = tied)
                                 SEXP xmu0SEXP, 
                                 SEXP xsigma0SEXP,
                                 SEXP item_sigmaSEXP,    // Prior covariance matrix for s_{jt} and m_{jt} (centered on sponsor's x_{it}) 
                                 SEXP omega2SEXP,
                                 SEXP rho_pSEXP,         // AR(1) coefficient in (-1,1). Use 1.0 for random walk.
                                 SEXP sig2_pSEXP,        // N x 1 innovation variance for propensity per legislator
                                 SEXP pmu0SEXP,          // N x 1 prior mean for p at first served period (often 0 and large).
                                 SEXP psigma0SEXP,       // N x 1 prior var for p at first served period (often 0 and large).
                                 SEXP threadsSEXP,
                                 SEXP verboseSEXP,
                                 SEXP maxitSEXP,
                                 SEXP threshSEXP,
                                 SEXP checkfreqSEXP
                                 ) {
  BEGIN_RCPP
    SEXP resultSEXP ;
  {
    Rcpp::RNGScope __rngScope ;
    Rcpp::traits::input_parameter<arma::mat>::type m_start(m_startSEXP);              // item motion 'm' start values
    Rcpp::traits::input_parameter<arma::mat>::type s_start(s_startSEXP);              // item status quo 's' start values
    Rcpp::traits::input_parameter<arma::mat>::type x_start(x_startSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type p_start(p_startSEXP) ;             // matrix of propensity starting values
    Rcpp::traits::input_parameter<arma::mat>::type y(ySEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type startlegis(startlegisSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type endlegis(endlegisSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type prevlegis(prevlegisSEXP) ;         // N x 1 column matrix of previous contiguous legislators (== 0 when no prior contiguous legislators).
    Rcpp::traits::input_parameter<arma::mat>::type bill_session(bill_sessionSEXP) ;
    Rcpp::traits::input_parameter<int>::type T(TSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type sponsor_index(sponsor_indexSEXP) ; // length J, 0-based row index of the sponsor MP for item j
    Rcpp::traits::input_parameter<arma::mat>::type anchor_group(anchor_groupSEXP) ;   // J x 1 (0 = singleton; >0 = tied)
    Rcpp::traits::input_parameter<arma::mat>::type xmu0(xmu0SEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type xsigma0(xsigma0SEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type item_sigma(item_sigmaSEXP) ;       // Prior covariance matrix for s_{jt} and m_{jt} (centered on sponsor's x_{it}) 
    Rcpp::traits::input_parameter<arma::mat>::type omega2(omega2SEXP) ;
    Rcpp::traits::input_parameter<double>::type rho_p(rho_pSEXP) ;                    // AR(1) coefficient in (-1,1). Use 1.0 for random walk.
    Rcpp::traits::input_parameter<arma::mat>::type sig2_p(sig2_pSEXP) ;               // N x 1 innovation variance for propensity per legislator
    Rcpp::traits::input_parameter<arma::mat>::type pmu0(pmu0SEXP) ;                   // N x 1 prior mean for p at first served period (often 0 and large).
    Rcpp::traits::input_parameter<arma::mat>::type psigma0(psigma0SEXP) ;             // N x 1 prior var for p at first served period (often 0 and large).
    Rcpp::traits::input_parameter<int>::type threads(threadsSEXP) ;
    Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP) ;
    Rcpp::traits::input_parameter<int>::type maxit(maxitSEXP) ;
    Rcpp::traits::input_parameter<double>::type thresh(threshSEXP) ;
    Rcpp::traits::input_parameter<int>::type checkfreq(checkfreqSEXP) ;
    
    Rcpp::List result = estimate_dynIRT(m_start,            // item motion 'm' start values
                                 s_start,                   // item status quo 's' start values
                                 x_start,
                                 p_start,                   // matrix of propensity starting values
                                 y, 
                                 startlegis,
                                 endlegis,
                                 bill_session,
                                 T,
                                 sponsor_index,             // length J, 0-based row index of the sponsor MP for item j
                                 anchor_group,              // J x 1 (0 = singleton; >0 = tied)
                                 xmu0,
                                 xsigma0,
                                 item_sigma,                // Prior covariance matrix for s_{jt} and m_{jt} (centered on sponsor's x_{it}) 
                                 omega2, 
                                 rho_p,                     // AR(1) coefficient in (-1,1). Use 1.0 for random walk.
                                 sig2_p,                    // N x 1 innovation variance for propensity per legislator
                                 pmu0,                      // N x 1 prior mean for p at first served period (often 0 and large).
                                 psigma0,                   // N x 1 prior var for p at first served period (often 0 and large).
                                 threads,
                                 verbose,
                                 maxit,
                                 thresh,
                                 checkfreq
                                 ) ;
    PROTECT(resultSEXP = Rcpp::wrap(result)) ;
  }
  UNPROTECT(1);
  return(resultSEXP) ;
  END_RCPP
    }
