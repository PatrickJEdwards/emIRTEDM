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
//#include "getVb2_dynIRT.h"
//#include "getEb2_dynIRT.h"
//#include "getVb_dynIRT.h"
//#include "getVa_dynIRT.h"
//#include "getEba_dynIRT.h"
//#include "getEbb_dynIRT.h"
#include "getLast_dynIRT.h"
#include "getX_dynIRT.h"
#include "getOnecol_dynIRT.h"
#include "checkConv_dynIRT.h"
#include "getP_dynIRT.h"          // function to update propensity parameters
#include "getMS_dynIRT.h"         // updates item m,s


using namespace Rcpp ;

List estimate_dynIRT(arma::mat m_start,   // J x 1: starting m
               arma::mat s_start,         // J x 1: starting s
               arma::mat x_start,         // N x T
               arma::mat p_start,         // N x T matrix of propensity starting values
               arma::mat y,               // N x J
               arma::mat startlegis,      // N x 1
               arma::mat endlegis,        // N x 1
               arma::mat bill_session,    // J x 1 (0..T-1)
               unsigned int T,
               arma::mat sponsor_index,   // J x 1 (row index of sponsor MP per item)
               arma::mat xmu0,            // N x 1
               arma::mat xsigma0,         // N x 1
               arma::mat item_sigma,      // 2x2 prior covariance for (m,s)
               arma::mat omega2,          // N x 1 (RW variance for x)
               double pmu = 0.0,          // propensity p_{it} prior mean value
               double psigma = 1.0,       // propensity p_{it} prior variance value
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
  
  
  // Basic checks
  if (psigma <= 0.0) Rcpp::stop("psigma must be > 0");
  if (x_start.n_rows != nN || x_start.n_cols != T) Rcpp::stop("x_start must be N x T");
  if (p_start.n_rows != nN || p_start.n_cols != T) Rcpp::stop("p_start must be N x T");
  if (m_start.n_rows != nJ || m_start.n_cols != 1) Rcpp::stop("m_start must be J x 1");
  if (s_start.n_rows != nJ || s_start.n_cols != 1) Rcpp::stop("s_start must be J x 1");
  if (sponsor_index.n_rows != nJ || sponsor_index.n_cols != 1) Rcpp::stop("sponsor_index must be J x 1");
  if (item_sigma.n_rows != 2 || item_sigma.n_cols != 2) Rcpp::stop("item_sigma must be 2 x 2");
  if (bill_session.n_rows != nJ || bill_session.n_cols != 1) Rcpp::stop("bill_session must be J x 1");
  if (startlegis.n_rows != nN || startlegis.n_cols != 1) Rcpp::stop("startlegis must be N x 1");
  if (endlegis.n_rows   != nN || endlegis.n_cols   != 1) Rcpp::stop("endlegis must be N x 1");
  
  
  // Check bill_session bounds and integer-ness:
  for (unsigned j = 0; j < nJ; ++j) {
    double t = bill_session(j,0);
    if (t < 0 || t >= (double)T) Rcpp::stop("bill_session(%u)=%g out of [0,T-1]", j, t);
    if (std::floor(t) != t) Rcpp::stop("bill_session(%u)=%g is not an integer index", j, t);
  }
    
  // Legislator service windows:
  for (unsigned i = 0; i < nN; ++i) {
    if (startlegis(i,0) < 0 || startlegis(i,0) >= (double)T) Rcpp::stop("startlegis(%u) out of range", i);
    if (endlegis(i,0)   < 0 || endlegis(i,0)   >= (double)T) Rcpp::stop("endlegis(%u) out of range", i);
    if (endlegis(i,0) < startlegis(i,0)) Rcpp::stop("endlegis < startlegis for i=%u", i);
  }
  
  // At least one item per period:
  for (unsigned t = 0; t < T; ++t) {
    bool any = false;
    for (unsigned j = 0; j < nJ; ++j) if (bill_session(j,0) == (double)t) { any = true; break; }
    if (!any) Rcpp::stop("No items found for period t=%u", t);
  }
  
  // At least one serving legislator per period:
  for (unsigned t = 0; t < T; ++t) {
    bool any = false;
    for (unsigned i = 0; i < nN; ++i) if (t >= startlegis(i,0) && t <= endlegis(i,0)) { any = true; break; }
    if (!any) Rcpp::stop("No serving legislators in period t=%u", t);
  }
  
    
  //// Initial "Current" Containers
  arma::mat curEystar(nN, nJ, arma::fill::zeros);
    
  // --- Items: primitives ---
  arma::mat curEm = m_start;   // J x 1
  arma::mat curEs = s_start;   // J x 1
    
  // Derived (for compatibility with downstream calls that still expect alpha/beta):
  arma::mat curEa(nJ,1);       // alpha = s^2 - m^2
  arma::mat curEb(nJ,1);       // beta  = 2(m - s)
    
  // Moments used by x-update: E[beta^2], E[beta*alpha]
  arma::mat curEbb(nJ,1);
  arma::mat curEba(nJ,1);
    
    
  // Optional posterior variances for (m,s) to be filled by the new item updater:
  arma::mat curVm(nJ,1, arma::fill::zeros);    // Var(m_jt)
  arma::mat curVs(nJ,1, arma::fill::zeros);    // Var(s_jt)
  arma::mat curCms(nJ,1, arma::fill::zeros);   // Cov(m_jt, s_jt)
    
  arma::mat Nlegis_session;	 // T x 1
  arma::mat legis_by_session; // list-like; dense matrix layout
  arma::cube curEx2x2(2, 2, T, arma::fill::zeros);
  //arma::cube curVb2(2, 2, T, arma::fill::zeros);   // (will be deprecated when items switch fully to m,s)
    
  //arma::mat curEb2(nJ, 2); // (will go away when we remove getEb2_dynIRT)
  arma::mat curVb;
  arma::mat curVa;
    
  arma::mat end_session;		
  //arma::mat ones_col;

  arma::mat curEx = x_start;                 // N x T
  arma::mat curVx(nN, T, arma::fill::zeros); // N x T
    
  arma::mat curEp = p_start;                 // N x T
  arma::mat curVp(nN, T, arma::fill::zeros); // N x T
    
  unsigned int i, j;

	/// Clean curEx outside service windows
	for(i=0; i < nN; i++){
	  for(j=0; j<T; j++){
	    if(j < startlegis(i,0)) curEx(i,j) = 0;
	    if(j > endlegis(i,0)) curEx(i,j) = 0;
	  }
	}
	// Clean curEp outside service windows
	for(i=0; i < nN; i++){
	  for(j=0; j<T; j++){
	    if(j < startlegis(i,0)) curEp(i,j) = 0;
	    if(j > endlegis(i,0)) curEp(i,j) = 0;
	  }
	}

	// Initialize derived alpha/beta and their moments from (m,s)
	for (j=0; j<nJ; ++j) {
	  const double m0 = curEm(j,0);
	  const double s0 = curEs(j,0);
	  curEb(j,0) = 2.0*(m0 - s0);
	  curEa(j,0) = s0*s0 - m0*m0;
	  
	  // Use prior Var(m,s) to seed E[β^2] and E[β·α]
	  const double vm0  = item_sigma(0,0);
	  const double vs0  = item_sigma(1,1);
	  const double cms0 = item_sigma(0,1);
	  
	  const double Eb = 2.0*(m0 - s0);
	  
	  // Var(β) = 4(Var(m)+Var(s) - 2 Cov(m,s))
	  const double Vb = 4.0 * (vm0 + vs0 - 2.0*cms0);
	  curEbb(j,0) = Vb + Eb*Eb;
	  
	  // E[β·α] for jointly Normal (m,s)
	  const double Em3  = m0*m0*m0 + 3.0*m0*vm0;
	  const double Es3  = s0*s0*s0 + 3.0*s0*vs0;
	  const double Ems2 = m0*(s0*s0 + vs0) + 2.0*s0*cms0;
	  const double Esm2 = s0*(m0*m0 + vm0) + 2.0*m0*cms0;
	  curEba(j,0) = 2.0 * (Ems2 - Em3 - Es3 + Esm2);
	  
	  // Seed ms-variances
	  curVm(j,0)  = vm0;
	  curVs(j,0)  = vs0;
	  curCms(j,0) = cms0;
	}
	
	// ---- Init "Old" containers to track for convergence ----
	arma::mat oldEm = curEm;   // track item m
	arma::mat oldEs = curEs;   // track item s
	arma::mat oldEa = curEa;   // track derived alpha (for backward-compat)
	arma::mat oldEb = curEb;   // track derived beta  (for backward-compat)
	arma::mat oldEx = curEx;   // legislator ideal points
	arma::mat oldEp = curEp;   // propensities


  // OpenMP Support
  #ifdef _OPENMP
  omp_set_num_threads(1) ;
  if (threads > 0) {
    omp_set_num_threads(threads) ;
    threadsused = omp_get_max_threads() ;
  }
  #endif

  // It turns out legis_by_session isn't necessary unless missing value in Ex is not 0. But only computed once, so no point changing it now
	legis_by_session = getLBS_dynIRT(startlegis, endlegis, T, nN);
  Nlegis_session = getNlegis_dynIRT(legis_by_session, T, nN);
	end_session = getLast_dynIRT(bill_session, T, nJ);
	//ones_col = getOnecol_dynIRT(startlegis, endlegis, T, nN);

	// ... inside estimate_dynIRT(...) before the main while-loop:
	//bool header_printed = false;
	
  // Main Loop Until Convergence
	while (counter < maxit) {
		
		counter++ ;
	  
	  
	  // 0) Keep alpha,beta synchronized with current (m,s)
	  for(j=0; j<nJ; j++){
	    double m = curEm(j,0), s = curEs(j,0);
	    curEb(j,0) = 2.0*(m - s);
	    curEa(j,0) = s*s - m*m;
	  }
		
		
		// 1) E[y*] (uses alpha,beta,p,x)
		getEystar_dynIRT(curEystar, curEa, curEb, curEx, curEp, y, bill_session, startlegis, endlegis,  nN, nJ);
	  if (!curEystar.is_finite()) Rcpp::stop("Eystar contains non-finite values after getEystar_dynIRT");
	  
	  
	  // 2) x-update (needs E[beta^2], E[beta*alpha]); we'll refresh them after the item step,
	  //    but for numerical stability keep last values for the very first x-update
	  getX_dynIRT(curEx, curVx, curEbb, omega2, curEb, curEystar, curEba,
               startlegis, endlegis, xmu0, xsigma0, T, nN, end_session, curEp);
	  if (!curEx.is_finite() || !curVx.is_finite()) Rcpp::stop("Ex/Vx non-finite after getX_dynIRT");
	  
	  
	  // 2.5) period-level sufficient stats for items (S0, Sx, Sx2)
	  getEx2x2_dynIRT(curEx2x2, curEx, curVx, legis_by_session, Nlegis_session, T);
	  
	  
	  // 3) (m,s) item update  — REPLACES getEx2x2/getVb2/getEb2
	  //    New function will update curEm,curEs and their posterior variances/covariances per item,
	  //    using: E[y*] - E[p] as the response and (E[x], Var[x]) as regressors, plus the sponsor prior.
	  //
	  //    Signature (to implement next):
	  //    getMS_dynIRT(curEm, curEs, curVm, curVs, curCms,
	  //                 curEystar, curEx, curEp,
	  //                 bill_session, sponsor_index,
	  //                 item_sigma,   // 2x2 prior covariance for (m,s) centered at (x_sponsor_t, 0)
	  //                 nJ, T);
	  //
	  getMS_dynIRT(curEm, curEs, curVm, curVs, curCms,
                curEystar, curEx, curEp,
                bill_session, sponsor_index,
                item_sigma, curEx2x2,
                startlegis, endlegis,
                nJ, nN, T);

	  // 4) Refresh derived alpha,beta and their moments from updated (m,s)
	  for(j=0; j<nJ; j++){
	    double m = curEm(j,0), s = curEs(j,0);
	    curEb(j,0) = 2.0*(m - s);
	    curEa(j,0) = s*s - m*m;
	    
	    // Moments E[beta^2], E[beta*alpha] using Gaussian ms-moments
	    double vm  = curVm(j,0), vs = curVs(j,0), cms = curCms(j,0);
	    // E[beta] and Var(beta)
	    double Eb   = curEb(j,0);
	    double Vb   = 4.0*(vm + vs - 2.0*cms);
	    double Eb2  = Vb + Eb*Eb;
	    curEbb(j,0) = Eb2;
	    
	    // E[alpha] = E[s^2] - E[m^2]
	    double Em2 = vm + m*m;
	    double Es2 = vs + s*s;
	    double Ea  = Es2 - Em2; // equals curEa(j,0) when using current means/vars
	    // E[beta*alpha] for jointly Normal (m,s)
	    double Em3 = m*m*m + 3.0*m*vm;
	    double Es3 = s*s*s + 3.0*s*vs;
	    double Ems2 = m*(s*s + vs) + 2.0*s*cms;
	    double Esm2 = s*(m*m + vm) + 2.0*m*cms;
	    double EbXa = 2.0*(Ems2 - Em3 - Es3 + Esm2);
	    curEba(j,0) = EbXa;
	  }
	  
    //getVb2_dynIRT(curVb2, curEx2x2, betasigma, T);
    //getEb2_dynIRT(curEb2, curEystar, curEx, curVb2, bill_session, betamu, betasigma, ones_col, nJ, curEp);
		//curEa = curEb2.col(0);
		//curEb = curEb2.col(1);
		// CHECK: ensure no NA/Inf slipped through 'getEb2_dynIRT(...)':
		//if (!curEa.is_finite() || !curEb.is_finite()) Rcpp::stop("alpha/beta non-finite after getEb2_dynIRT");
		//curEba = getEba_dynIRT(curEa,curEb,curVb2,bill_session,nJ);
		//curEbb = getEbb_dynIRT(curEb,curVb2,bill_session,nJ);
		
		
		
		// 5) propensity (non-dynamic i.i.d. Normal prior) — unchanged
		getP_dynIRT(curEp, curVp, curEystar, curEa, curEb, curEx,
              bill_session, startlegis, endlegis, pmu, psigma, T, nN, nJ);
	  if (!curEp.is_finite()) Rcpp::stop("Ep contains non-finite values after getP_dynIRT");
	  
	
	
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
		oldEp = curEp;
		oldEa = curEa;   // keep derived for compatibility with existing checkConv
		oldEb = curEb;
		oldEm = curEm;   // NEW
		oldEs = curEs;   // NEW

	}
  // LOOP ENDING HERE

  //curVb = getVb_dynIRT(curVb2, bill_session, nJ);  // TODO: will be removed
	//curVa = getVa_dynIRT(curVb2, bill_session, nJ);  // TODO: will be removed

	// After convergence: compute Var(beta) and Var(alpha) directly from (m,s)
	// Var(beta) = 4(Var(m)+Var(s) - 2 Cov(m,s))
	// Var(alpha) = Var(s^2) + Var(m^2) - 2 Cov(m^2, s^2)
	//   with for jointly normal (m,s):
	//   Var(m^2)   = 2*vm^2 + 4*m^2*vm
	//   Var(s^2)   = 2*vs^2 + 4*s^2*vs
	//   Cov(m^2,s^2) = 2*cms^2 + 4*m*s*cms
	
	curVb.set_size(nJ, 1);
	curVa.set_size(nJ, 1);
	
	for (unsigned j = 0; j < nJ; ++j) {
	  const double m   = curEm(j,0);
	  const double s   = curEs(j,0);
	  const double vm  = curVm(j,0);   // Var(m)
	  const double vs  = curVs(j,0);   // Var(s)
	  const double cms = curCms(j,0);  // Cov(m,s)
	  
	  // Var(beta)
	  double var_beta = 4.0 * (vm + vs - 2.0 * cms);
	  
	  // Var(alpha)
	  const double var_m2   = 2.0 * vm * vm + 4.0 * m * m * vm;
	  const double var_s2   = 2.0 * vs * vs + 4.0 * s * s * vs;
	  const double cov_m2s2 = 2.0 * cms * cms + 4.0 * m * s * cms;
	  double var_alpha = var_s2 + var_m2 - 2.0 * cov_m2s2;
	  
	  // numerical safety (round tiny negatives to 0)
	  if (var_beta  < 0 && var_beta  > -1e-12)  var_beta  = 0.0;
	  if (var_alpha < 0 && var_alpha > -1e-12) var_alpha = 0.0;
	  
	  curVb(j,0) = var_beta;
	  curVa(j,0) = var_alpha;
	}
	
	
	
  // 	Rcout << "\n Completed after " << counter << " iterations..." << std::endl ;

    List ret ;
    List means ;
    List vars ;
    List runtime ;

    means["x"]     = curEx;
    means["alpha"] = curEa;
    means["beta"]  = curEb;
    means["p"]     = curEp;
    means["m"]     = curEm;
    means["s"]     = curEs;
	
	  vars["x"]      = curVx;
	  vars["alpha"]  = curVa;   // Var(alpha) from (m,s)
	  vars["beta"]   = curVb;   // Var(beta)  from (m,s)
	  vars["p"]      = curVp;
	  vars["var_m"]  = curVm;   // optional: expose Var(m)
	  vars["var_s"]  = curVs;   // optional: expose Var(s)
	  vars["cov_ms"] = curCms;  // optional: expose Cov(m,s)
    
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
