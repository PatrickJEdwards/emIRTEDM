// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#define DEBUG false

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include <iomanip>
#include "getEystar_dynIRT.h"
#include "getLBS_dynIRT.h"
#include "getNlegis_dynIRT.h"
#include "getEx2x2_dynIRT.h"
#include "getVb2_dynIRT.h"
#include "getEb2_dynIRT.h"
#include "getVb_dynIRT.h"
#include "getVa_dynIRT.h"
#include "getEba_dynIRT.h"
#include "getEbb_dynIRT.h"
#include "getLast_dynIRT.h"
#include "getX_dynIRT.h"
#include "getOnecol_dynIRT.h"
#include "checkConv_dynIRT.h"
#include "getP_dynIRT.h"          // NEW: function to update propensity parameters


using namespace Rcpp ;

List estimate_dynIRT(arma::mat alpha_start,
               arma::mat beta_start,
               arma::mat x_start,
               arma::mat p_start,         // matrix of propensity starting values
               arma::mat y,
               arma::mat startlegis,
               arma::mat endlegis,
               arma::mat bill_session,
               unsigned int T,
               arma::mat xmu0,
               arma::mat xsigma0,
               arma::mat betamu,
               arma::mat betasigma,
               arma::mat omega2,
               double pmu = 0.0,         // propensity p_{it} prior mean value
               double psigma = 1.0,      // propensity p_{it} prior variance value
               unsigned int threads = 0,
               bool verbose = true,
               unsigned int maxit = 2500,
               double thresh = 1e-6,
               unsigned int checkfreq = 50
               ) {

    //// Data Parameters
    unsigned int nJ = y.n_cols ;
    unsigned int nN = y.n_rows ;
   
    //// Admin
    unsigned int threadsused = 0 ;
	  int convtype=1;
    unsigned int counter = 0 ;
    int isconv = 0;
    
    
    // Check Input Data:
    if (psigma <= 0.0) Rcpp::stop("psigma must be > 0");
    if (x_start.n_rows != nN || x_start.n_cols != T) Rcpp::stop("x_start must be N x T");
    if (p_start.n_rows != nN || p_start.n_cols != T) Rcpp::stop("p_start must be N x T");
    if (alpha_start.n_rows != nJ || alpha_start.n_cols != 1) Rcpp::stop("alpha_start must be J x 1");
    if (beta_start.n_rows  != nJ || beta_start.n_cols  != 1) Rcpp::stop("beta_start must be J x 1");
    if (bill_session.n_rows != nJ || bill_session.n_cols != 1) Rcpp::stop("bill_session must be J x 1");
    if (startlegis.n_rows != nN || startlegis.n_cols != 1) Rcpp::stop("startlegis must be N x 1");
    if (endlegis.n_rows   != nN || endlegis.n_cols   != 1) Rcpp::stop("endlegis must be N x 1");
    
    // Check bill_session bounds and integer-ness:
    for (unsigned j = 0; j < nJ; ++j) {
      double t = bill_session(j,0);
      if (t < 0 || t >= (double)T) Rcpp::stop("bill_session(%u)=%g out of [0,T-1]", j, t);
      if (std::floor(t) != t) Rcpp::stop("bill_session(%u)=%g is not an integer index", j, t);
    }
    
    // Check legislator service windows:
    for (unsigned i = 0; i < nN; ++i) {
      if (startlegis(i,0) < 0 || startlegis(i,0) >= (double)T) Rcpp::stop("startlegis(%u) out of range", i);
      if (endlegis(i,0)   < 0 || endlegis(i,0)   >= (double)T) Rcpp::stop("endlegis(%u) out of range", i);
      if (endlegis(i,0) < startlegis(i,0)) Rcpp::stop("endlegis < startlegis for i=%u", i);
    }
    
    // Check that there's at least one item per period:
    for (unsigned t = 0; t < T; ++t) {
      bool any = false;
      for (unsigned j = 0; j < nJ; ++j) if (bill_session(j,0) == (double)t) { any = true; break; }
      if (!any) Rcpp::stop("No items found for period t=%u", t);
    }
    
    // Check that there's at least one legislator serving per period:
    for (unsigned t = 0; t < T; ++t) {
      bool any = false;
      for (unsigned i = 0; i < nN; ++i) if (t >= startlegis(i,0) && t <= endlegis(i,0)) { any = true; break; }
      if (!any) Rcpp::stop("No serving legislators in period t=%u", t);
    }
    
    
    //// Initial "Current" Containers
    arma::mat curEystar(nN, nJ, arma::fill::zeros);
    arma::mat curEa = alpha_start;
    arma::mat curEb = beta_start;
	  arma::mat Nlegis_session;	// T x 1, each element has N number of legislators for session t
	  arma::mat legis_by_session;	// T rows, each row has vector of legislators in session
	  arma::cube curEx2x2(2, 2, T, arma::fill::zeros);
	  arma::cube curVb2(2, 2, T, arma::fill::zeros);
    arma::mat curEb2(nJ, 2) ;
    arma::mat curVb;
    arma::mat curVa;
	  arma::mat end_session;		// Filled by getLast()
	  arma::mat ones_col;

    arma::mat curEx = x_start; // (nNxT) matrix
	    // Filling zeroes important for Vx, as getEx2x2() assumes Vx=0 for missing legislators for that period
	    // Since Vx never gets updated for missing legislators, only need to do this once
    arma::mat curVx(nN, T, arma::fill::zeros);
    
    // NEW: initial "current" containers for propensity parameters
    arma::mat curEp = p_start;                // nN x T matrix of propensity mean start values
    arma::mat curVp(nN, T, arma::fill::zeros); // nN x T matrix of propensity variances (initially set to zero)
    
	unsigned int i, j;

	// Clean curEx, setting all values of x_{it} that are not estimated (i.e. before startlegis and after endlegis) to 0
	// This guarantees that for final output, ideal points not estimated are output as 0 regardless of starting value
	for(i=0; i < nN; i++){
		for(j=0; j<T; j++){
			if(j < startlegis(i,0)) curEx(i,j) = 0;
			if(j > endlegis(i,0)) curEx(i,j) = 0;
		}
	}
	
	// NEW: Clean curEp outside service windows (keep means 0 there)
	for(i=0; i < nN; i++){
	  for(j=0; j<T; j++){
	    if(j < startlegis(i,0)) curEp(i,j) = 0;
	    if(j > endlegis(i,0)) curEp(i,j) = 0;
	  }
	}

	arma::mat curEbb(nJ,1);
	arma::mat curEba(nJ,1);
	for(j=0; j<nJ; j++){
		curEbb(j,0) = beta_start(j,0)*beta_start(j,0);
		curEba(j,0) = alpha_start(j,0)*beta_start(j,0);
	}
	
    
  //// Init "Old" Containers to track for convergence
  arma::mat oldEa = alpha_start;
  arma::mat oldEb = beta_start;
  arma::mat oldEx = curEx;
  arma::mat oldEp = curEp; // NEW: track old propensities for convergence


  // OpenMP Support
  #ifdef _OPENMP
  omp_set_num_threads(1) ;
  if (threads > 0) {
    omp_set_num_threads(threads) ;
    threadsused = omp_get_max_threads() ;
  }
  #endif

  // It turns out legis_by_session isn't necessary unless missing value in Ex is not 0
  // But only computed once, so no point changing it now
	legis_by_session = getLBS_dynIRT(startlegis, endlegis, T, nN);
  Nlegis_session = getNlegis_dynIRT(legis_by_session, T, nN);
	end_session = getLast_dynIRT(bill_session, T, nJ);
	ones_col = getOnecol_dynIRT(startlegis, endlegis, T, nN);

  // Main Loop Until Convergence
	while (counter < maxit) {
		
		counter++ ;
		  
		  
		// CHANGED: E[y*] now depends on p_{it} as well. New signature will accept curEp
		getEystar_dynIRT(
      curEystar,
      curEa, curEb,
      curEx, curEp,      // <- NEW: added curEp to function inputs
      y,
      bill_session,
      startlegis, endlegis, 
      nN, nJ
    );
		
    // CHECK: ensure no NA/Inf slipped through 'getEystar_dynIRT(...)':
    if (!curEystar.is_finite()) Rcpp::stop("Eystar contains non-finite values after getEystar_dynIRT");
    
    
    
    // NEW: propensity update (non-dynamic, per-time sum-to-zero inside)
    // Uses residuals r = E[y*] - E[alpha] - E[beta] E[x]
    // Prior N(pmu, psigma); centers within each time period.
    getP_dynIRT(
      curEp, curVp,            // updated in place
      curEystar,               // E[y*]
      curEa, curEb, curEx,     // alpha, beta, x
      bill_session,            // to know which items live in each t
      startlegis, endlegis,
      pmu, psigma,             // Prior mean and variance of propensity parameters 
      T, nN, nJ
    );
    
    // CHECK: ensure no NA/Inf slipped through 'getP_dynIRT(...)':
    if (!curEp.is_finite()) Rcpp::stop("Ep contains non-finite values after getP_dynIRT");
  
  
		
		// CHANGED: x-update uses de-propensitied pseudo-observations
		// New signature will accept curEp
		getX_dynIRT(
		  curEx, curVx,
		  curEbb, omega2,
		  curEb, curEystar, curEba,
		  startlegis, endlegis,
		  xmu0, xsigma0,
		  T, nN, end_session,
		  curEp                       // <-- NEW
		);
		
		// CHECK: ensure no NA/Inf slipped through 'getX_dynIRT(...)':
		if (!curEx.is_finite() || !curVx.is_finite()) Rcpp::stop("Ex/Vx non-finite after getX_dynIRT");
		

		
		getEx2x2_dynIRT(curEx2x2,curEx,curVx,legis_by_session,Nlegis_session,T);
    getVb2_dynIRT(curVb2, curEx2x2, betasigma, T);
    	
    	
    	
    // CHANGED: item update uses y^dagger = E[y*] - E[p]
    // New signature will accept curEp
    getEb2_dynIRT(
      curEb2,
      curEystar, curEx, curVb2,
      bill_session,
      betamu, betasigma,
      ones_col,
      nJ,
      curEp                      // <-- NEW
    );
  
		curEa = curEb2.col(0);
		curEb = curEb2.col(1);
		
		// CHECK: ensure no NA/Inf slipped through 'getX_dynIRT(...)':
		if (!curEa.is_finite() || !curEb.is_finite()) Rcpp::stop("alpha/beta non-finite after getEb2_dynIRT");
		
		

		curEba = getEba_dynIRT(curEa,curEb,curVb2,bill_session,nJ);
		curEbb = getEbb_dynIRT(curEb,curVb2,bill_session,nJ);

		
		
		
		// ---- Progress + Convergence (table-style output from new checkConv) ----
		
		
		// Only compute deviations once per iter; also use them for printing
		bool do_check = (counter > 2);
		bool do_print = (verbose && (counter % checkfreq == 0));
		
		
		if (do_check || do_print) {
		  // Returns: list(devEx, devEb, devEa, devEp, check)
		  Rcpp::List conv = checkConv_dynIRT(
		    oldEx, curEx, oldEb, curEb, oldEa, curEa, oldEp, curEp, thresh, convtype
		  );
		  
		  double devEx = Rcpp::as<double>(conv["devEx"]);
		  double devEb = Rcpp::as<double>(conv["devEb"]);
		  double devEa = Rcpp::as<double>(conv["devEa"]);
		  double devEp = Rcpp::as<double>(conv["devEp"]);
		  bool   check = Rcpp::as<bool>(conv["check"]);
		  
		  // Nicely aligned progress table
		  if (do_print) {
		    static bool header_printed = false;
		    if (!header_printed) {
		      Rcout << std::left
              << std::setw(10) << "Iteration"
              << std::setw(14) << "Dev. Ex"
              << std::setw(14) << "Dev. Eb"
              << std::setw(14) << "Dev. Ea"
              << std::setw(14) << "Dev. Ep"
              << "\n";
		      header_printed = true;
		    }
		    Rcout << std::left << std::setw(10) << counter
            << std::scientific << std::setprecision(3)
            << std::setw(14) << devEx
            << std::setw(14) << devEb
            << std::setw(14) << devEa
            << std::setw(14) << devEp
            << std::defaultfloat << "\n";
		  }
		  
		  // Use the logical element of conv_check_output to decide to break
		  if (do_check && check) {
		    isconv = 1;    // keep runtime info consistent
		    break;
		  }
		}
		
		
		// Still respect user interrupts each print interval
		if (do_print) {
		  R_CheckUserInterrupt();
		}

		
		
		// Update Old Values If Not Converged
		oldEx = curEx;
		oldEb = curEb;
		oldEa = curEa;
		oldEp = curEp; // NEW

	}
// LOOP ENDING HERE

	// Only needed after convergence
  curVb = getVb_dynIRT(curVb2, bill_session, nJ);
  curVa = getVa_dynIRT(curVb2, bill_session, nJ);

  // 	Rcout << "\n Completed after " << counter << " iterations..." << std::endl ;

    List ret ;
    List means ;
    List vars ;
    List runtime ;

    means["x"]     = curEx;
    means["alpha"] = curEa;
    means["beta"]  = curEb;
    means["p"]     = curEp;     // NEW
	
    vars["x"]      = curVx;
    vars["alpha"]  = curVa;
    vars["beta"]   = curVb;
    vars["p"]      = curVp;     // NEW
    
    runtime["iters"]     = counter;
    runtime["conv"]      = isconv;
    runtime["threads"]   = threadsused;
    runtime["tolerance"] = thresh;

    ret["means"]   = means;
    ret["vars"]    = vars;
    ret["runtime"] = runtime;

    ret["N"] = nN;
    ret["J"] = nJ;
    ret["T"] = T;

    return(ret);
}
