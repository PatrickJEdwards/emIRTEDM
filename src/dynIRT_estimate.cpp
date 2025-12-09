#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "estimate_dynIRT.h"

RcppExport SEXP dynIRT_estimate(SEXP m_startSEXP,        // item motion 'm' start values
                                 SEXP s_startSEXP,       // item status quo 's' start values
                                 //SEXP alpha_startSEXP,
                                 //SEXP beta_startSEXP,
                                 SEXP x_startSEXP,
                                 SEXP p_startSEXP,       // propensity p_{it} start values
                                 SEXP ySEXP,
                                 SEXP startlegisSEXP,
                                 SEXP endlegisSEXP,
                                 SEXP bill_sessionSEXP,
                                 SEXP TSEXP,
                                 SEXP sponsor_indexSEXP, // length J, 0-based row index of the sponsor MP for item j
                                 SEXP xmu0SEXP, 
                                 SEXP xsigma0SEXP,
                                 //SEXP item_muSEXP,     
                                 SEXP item_sigmaSEXP,    // Prior covariance matrix for s_{jt} and m_{jt} (centered on sponsor's x_{it}) 
                                 //SEXP betamuSEXP, 
                                 //SEXP betasigmaSEXP, 
                                 SEXP omega2SEXP,
                                 SEXP pmuSEXP,           // propensity p_{it} prior mean value
                                 SEXP psigmaSEXP,        // propensity p_{it} prior variance value
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
    //Rcpp::traits::input_parameter<arma::mat>::type alpha_start(alpha_startSEXP);
    //Rcpp::traits::input_parameter<arma::mat>::type beta_start(beta_startSEXP);
    Rcpp::traits::input_parameter<arma::mat>::type x_start(x_startSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type p_start(p_startSEXP) ;             // matrix of propensity starting values
    Rcpp::traits::input_parameter<arma::mat>::type y(ySEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type startlegis(startlegisSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type endlegis(endlegisSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type bill_session(bill_sessionSEXP) ;
    Rcpp::traits::input_parameter<int>::type T(TSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type sponsor_index(sponsor_indexSEXP) ; // length J, 0-based row index of the sponsor MP for item j
    Rcpp::traits::input_parameter<arma::mat>::type xmu0(xmu0SEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type xsigma0(xsigma0SEXP) ;
    //Rcpp::traits::input_parameter<arma::mat>::type item_mu(item_muSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type item_sigma(item_sigmaSEXP) ;       // Prior covariance matrix for s_{jt} and m_{jt} (centered on sponsor's x_{it}) 
    //Rcpp::traits::input_parameter<arma::mat>::type betamu(betamuSEXP) ;
    //Rcpp::traits::input_parameter<arma::mat>::type betasigma(betasigmaSEXP) ;
    Rcpp::traits::input_parameter<arma::mat>::type omega2(omega2SEXP) ;
    Rcpp::traits::input_parameter<double>::type pmu(pmuSEXP) ;                        // propensity p_{it} prior mean value
    Rcpp::traits::input_parameter<double>::type psigma(psigmaSEXP) ;                  // propensity p_{it} prior variance value
    Rcpp::traits::input_parameter<int>::type threads(threadsSEXP) ;
    Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP) ;
    Rcpp::traits::input_parameter<int>::type maxit(maxitSEXP) ;
    Rcpp::traits::input_parameter<double>::type thresh(threshSEXP) ;
    Rcpp::traits::input_parameter<int>::type checkfreq(checkfreqSEXP) ;
    
    Rcpp::List result = estimate_dynIRT(m_start, // item motion 'm' start values
                                 s_start         // item status quo 's' start values
                                 //alpha_start,
                                 //beta_start,
                                 x_start,
                                 p_start,        // matrix of propensity starting values
                                 y, 
                                 startlegis,
                                 endlegis,
                                 bill_session,
                                 T,
                                 sponsor_index,  // length J, 0-based row index of the sponsor MP for item j
                                 xmu0,
                                 xsigma0,
                                 item_sigma,     // Prior covariance matrix for s_{jt} and m_{jt} (centered on sponsor's x_{it}) 
                                 //betamu,
                                 //betasigma, 
                                 omega2, 
                                 pmu,            // propensity p_{it} prior mean value
                                 psigma,         // propensity p_{it} prior variance value
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
