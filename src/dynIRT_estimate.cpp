#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "estimate_dynIRT.h"

RcppExport SEXP dynIRT_estimate(SEXP alpha_startSEXP,
                                 SEXP beta_startSEXP,
                                 SEXP x_startSEXP,
                                 SEXP p_startSEXP,     // propensity p_{it} start values
                                 SEXP ySEXP,
                                 SEXP startlegisSEXP,
                                 SEXP endlegisSEXP,
                                 SEXP bill_sessionSEXP,
                                 SEXP TSEXP,
                                 SEXP xmu0SEXP, 
                                 SEXP xsigma0SEXP,
                                 SEXP betamuSEXP, 
                                 SEXP betasigmaSEXP, 
                                 SEXP omega2SEXP,
                                 SEXP pmuSEXP,        // propensity p_{it} prior mean value
                                 SEXP psigmaSEXP,     // propensity p_{it} prior variance value
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
    Rcpp::traits::input_parameter<arma::mat>::type alpha_start(alpha_startSEXP);
    Rcpp::traits::input_parameter<arma::mat>::type beta_start(beta_startSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type x_start(x_startSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type p_start(p_startSEXP) ; // matrix of propensity starting values
    Rcpp::traits::input_parameter<arma::mat>::type y(ySEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type startlegis(startlegisSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type endlegis(endlegisSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type bill_session(bill_sessionSEXP) ;
    Rcpp::traits::input_parameter<int>::type T(TSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type xmu0(xmu0SEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type xsigma0(xsigma0SEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type betamu(betamuSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type betasigma(betasigmaSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type omega2(omega2SEXP) ;
    Rcpp::traits::input_parameter<double>::type pmu(pmuSEXP) ;       // propensity p_{it} prior mean value
    Rcpp::traits::input_parameter<double>::type psigma(psigmaSEXP) ; // propensity p_{it} prior variance value
    Rcpp::traits::input_parameter<int>::type threads(threadsSEXP) ;
    Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP) ;
    Rcpp::traits::input_parameter<int>::type maxit(maxitSEXP) ;
    Rcpp::traits::input_parameter<double>::type thresh(threshSEXP) ;
    Rcpp::traits::input_parameter<int>::type checkfreq(checkfreqSEXP) ;
    
    Rcpp::List result = estimate_dynIRT(alpha_start,
                                 beta_start,
                                 x_start,
                                 p_start,      // matrix of propensity starting values
                                 y, 
                                 startlegis,
                                 endlegis,
                                 bill_session,
                                 T,
                                 xmu0,
                                 xsigma0,
                                 betamu,
                                 betasigma, 
                                 omega2, 
                                 pmu,         // propensity p_{it} prior mean value
                                 psigma,      // propensity p_{it} prior variance value
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
