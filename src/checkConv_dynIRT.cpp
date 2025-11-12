// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>

using namespace Rcpp;

int checkConv_dynIRT(const arma::mat &oldEx,
                     const arma::mat &curEx,
                     const arma::mat &oldEb,
                     const arma::mat &curEb,
                     const arma::mat &oldEa,
                     const arma::mat &curEa,
                     const arma::mat &oldEp,   // NEW: propensities (previous)
                     const arma::mat &curEp,   // NEW: propensities (current)
                     double thresh,
                     int convtype) {

  double devEx = 100.0;
  double devEa = 100.0;
  double devEb = 100.0;
  double devEp = 100.0;   // NEW

  
  // Strip zeros to ignore out-of-service cells for x
  arma::vec oldEx_vec = vectorise(oldEx);
  arma::vec curEx_vec = vectorise(curEx);
  arma::vec oldEx_stripped = oldEx_vec(arma::find(oldEx_vec != 0));
  arma::vec curEx_stripped = curEx_vec(arma::find(curEx_vec != 0));
  
  // Strip zeros for p (same rationale as x)
  arma::vec oldEp_vec = vectorise(oldEp);
  arma::vec curEp_vec = vectorise(curEp);
  arma::vec oldEp_stripped = oldEp_vec(arma::find(oldEp_vec != 0));
  arma::vec curEp_stripped = curEp_vec(arma::find(curEp_vec != 0));
  
  // One-time informative warning if Ep is effectively all zeros after stripping
  const bool ep_empty = (oldEp_stripped.n_elem == 0u) || (curEp_stripped.n_elem == 0u);
  if (ep_empty) {
    static bool warned = false;
    if (!warned) {
      Rcpp::warning("checkConv_dynIRT: propensity means Ep are all zero after masking "
                      "out-of-service cells; verify getP_dynIRT and bill_session wiring. "
                      "Proceeding with Ep convergence metric set to zero.");
      warned = true;
      //Rcpp::stop("checkConv_dynIRT: Ep is all zeros after masking; check getP_dynIRT and bill_session."); // Use instead of the warning if I want a strict error message + stop code running
    }
    devEp = 0.0; // don’t block convergence; we warned the user
  }
  
  
  if (convtype == 1) {
    // correlation distance (1 - corr)
    devEx = 1 - (cor(oldEx_stripped, curEx_stripped)).min();
    devEb = 1 - (cor(oldEb,          curEb         )).min();
    devEa = 1 - (cor(oldEa,          curEa         )).min();
    devEp = 1 - (cor(oldEp_stripped, curEp_stripped)).min(); // NEW
  }
  
  if (convtype == 2) {
    // maximum absolute deviation
    devEx = (abs(curEx_stripped - oldEx_stripped)).max();
    devEb = (abs(curEb - oldEb)).max();
    devEa = (abs(curEa - oldEa)).max();
    if (!ep_empty) {
      devEp = (abs(curEp_stripped - oldEp_stripped)).max(); // NEW
    }
  }
  
  bool check = (devEx < thresh) & (devEb < thresh) & (devEa < thresh) & (devEp < thresh) ;
  
  // Return as an Rcpp list (with names, recommended)
  return Rcpp::List::create(
    _["devEx"] = devEx,
    _["devEb"] = devEb,
    _["devEa"] = devEa,
    _["devEp"] = devEp,
    _["check"] = check
  );
  
  //if( (devEx < thresh) & (devEb < thresh) & (devEa < thresh) & (devEp < thresh)) return(1);
  //return(0) ;

}
