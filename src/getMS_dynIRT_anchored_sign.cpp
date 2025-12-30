#include "getMS_dynIRT_anchored_sign.h"

#include <RcppArmadillo.h>
#include <limits>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>

using arma::mat; using arma::vec; using arma::uword; using arma::ivec; using arma::cube;

// ---- Smooth box barrier helpers ----
//static inline double softplus(double z){ return std::log1p(std::exp(z)); }
static inline double softplus(double z){
  // stable log(1+exp(z))
  if (z > 50.0) return z;            // log(1+exp(z)) ~ z
  if (z < -50.0) return std::exp(z); // log(1+exp(z)) ~ exp(z)
  return std::log1p(std::exp(z));
}

//static inline double sigmoid (double z){ return 1.0 / (1.0 + std::exp(-z)); }
static inline double sigmoid(double z){
  // stable logistic
  if (z >= 0.0) {
    double ez = std::exp(-z);
    return 1.0 / (1.0 + ez);
  } else {
    double ez = std::exp(z);
    return ez / (1.0 + ez);
  }
}



static inline double sq(double x){ return x*x; }

// Stable log(exp(b) - exp(a)) where inputs are logs
static inline double log_diff_exp(double log_b, double log_a){
  if (log_b < log_a) std::swap(log_b, log_a);
  const double x = std::exp(log_a - log_b);
  if (x >= 1.0) return -std::numeric_limits<double>::infinity();
  return log_b + std::log1p(-x);
}

// Univariate truncated-Normal moments for truncation to [L,U]
static inline std::pair<double,double>
  trunc_box_scalar(double mu, double var, double L, double U){
    const double VAR_FLOOR = 1e-12;
    var = std::max(var, VAR_FLOOR);
    
    if (!(U > L)) {
      double m = std::min(U, std::max(L, mu));
      return { m, VAR_FLOOR };
    }
    
    const double sd = std::sqrt(var);
    const double a  = (L - mu)/sd;
    const double b  = (U - mu)/sd;
    
    const double log_phi_a = R::dnorm(a, 0.0, 1.0, /*log*/true);
    const double log_phi_b = R::dnorm(b, 0.0, 1.0, /*log*/true);
    const double log_Phi_a = R::pnorm(a, 0.0, 1.0, /*lower*/true,  /*log*/true);
    const double log_Phi_b = R::pnorm(b, 0.0, 1.0, /*lower*/true,  /*log*/true);
    
    const double logZ = log_diff_exp(log_Phi_b, log_Phi_a);
    if (!std::isfinite(logZ)) {
      double m = std::min(U, std::max(L, mu));
      return { m, VAR_FLOOR };
    }
    
    const double r_a = std::exp(log_phi_a - logZ);
    const double r_b = std::exp(log_phi_b - logZ);
    
    const double mean = mu + sd * (r_a - r_b);
    const double term = (a * r_a - b * r_b);
    double variance   = var * (1.0 + term - (r_a - r_b)*(r_a - r_b));
    if (!(variance > 0.0) || !std::isfinite(variance)) variance = VAR_FLOOR;
    
    double mean_box = std::min(U, std::max(L, mean));
    return { mean_box, variance };
  }

static inline double inv_softplus(double y){
  const double EPS = 1e-12;
  y = std::max(y, EPS);
  // log(expm1(y)) is stable for small y
  return std::log(std::expm1(y));
}

// Map unconstrained u -> constrained d plus first/second derivatives
// bs = +1 : d =  softplus(u) >= 0
// bs = -1 : d = -softplus(u) <= 0
// bs =  0 : d = u (unconstrained)
static inline void d_from_u(int bs, double u, double &d, double &d1, double &d2){
  if (bs == 0){
    d  = u;
    d1 = 1.0;
    d2 = 0.0;
    return;
  }
  const double sp  = softplus(u);
  const double sig = sigmoid(u);
  const double curv = sig * (1.0 - sig); // derivative of sigmoid
  
  if (bs == 1){
    d  = sp;
    d1 = sig;
    d2 = curv;
  } else { // bs == -1
    d  = -sp;
    d1 = -sig;
    d2 = -curv;
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// NEW: group-anchored updater. Items with the same positive group id share one
// (m,s). Singletons (group==0) use the original per-item logic verbatim.
// Tie-break for the group's prior mean of m: among items occurring in the
// earliest session where the group appears, pick the item with the smallest
// edm_index (i.e., smallest j) to choose the sponsor used for mu_m.
// ─────────────────────────────────────────────────────────────────────────────
void getMS_dynIRT_anchored_sign(
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
    const mat& beta_sign,
    bool prior_use_first_occurrence,
    unsigned int newton_maxit, 
    double newton_tol, 
    double ridge
){
  // same hard box / barrier you use in the per-item solver
  const double MS_MIN = -500.0, MS_MAX = 500.0;
  const double LAMBDA_BAR = 1e-2;
  
  if (!item_sigma.is_sympd()) Rcpp::stop("item_sigma must be SPD");
  mat Lambda = arma::inv_sympd(item_sigma); // 2x2
  
  // Validate beta_sign dims/values
  if (beta_sign.n_rows != nJ || beta_sign.n_cols != 1) {
    Rcpp::stop("beta_sign must be J x 1. Got %d x %d, expected %d x 1.",
               (int)beta_sign.n_rows, (int)beta_sign.n_cols, (int)nJ);
  }
  for (uword j = 0; j < nJ; ++j) {
    const double v = beta_sign(j,0);
    const int sgn = (int)std::llround(v);
    if (!std::isfinite(v) || std::fabs(v - (double)sgn) > 1e-8 || (sgn != -1 && sgn != 0 && sgn != 1)) {
      Rcpp::stop("beta_sign[%d]=%f invalid. Must be -1, 0, or 1.", (int)j+1, v);
    }
  }
  
  
  // Build group -> item index lists (consider >0 as tied; 0/-1 singletons ignored here)
  arma::ivec gids = arma::unique(anchor_group.elem( arma::find(anchor_group > 0) ));
  // If no groups, we’ll still run the singleton block below.
  if (gids.is_empty()) { /* fall through */ }
  
  // Parallel over groups (completely independent)
#pragma omp parallel for schedule(static)
  for (int gg = 0; gg < (int)gids.n_elem; ++gg) {
    int gid = gids(gg);
    
    // collect item indices in this group
    std::vector<uword> items;
    items.reserve(8);
    for (uword j = 0; j < nJ; ++j) {
      if (anchor_group((int)j) == gid) items.push_back(j);
    }
    if (items.size() <= 1) continue;
      
    // pooled sufficient stats across all items in the group
    double S0  = 0.0, Sx  = 0.0, Sx2 = 0.0;
    double Sy  = 0.0, Syx = 0.0;
    
    
    
    
    // choose prior mean for m: sponsor_x from earliest session; 
    // if multiple items tie on that earliest session, break the tie by the 
    // smallest EDM index (i.e., smallest j in 0..nJ-1). 
    int j_ref = -1; int t_ref = std::numeric_limits<int>::max(); 
    
    for (uword k = 0; k < items.size(); ++k) {
      const int j = static_cast<int>(items[k]);            // item's column index (0-based)
      const int t = static_cast<int>(bill_session(j, 0));  // item's session (0..T-1)
      if (t < t_ref) {
        t_ref = t;
        j_ref = j;
      } else if (t == t_ref && j < j_ref) {
        // tie on earliest session: choose smallest EDM index
        j_ref = j;
      }
    }
    
    // safety checks (should never trigger in valid input) 
    if (j_ref < 0 || t_ref < 0 || t_ref >= static_cast<int>(T)) {
      Rcpp::stop("anchor-group tie-break failed: invalid j_ref/t_ref (group %d)", gid);
    }
    
    // sponsor prior center for m from the selected reference item 
    int si = static_cast<int>(sponsor_index(j_ref, 0)) - 1; // 1-based -> 0-based 
    if (si < 0 || si >= static_cast<int>(nN)) {
      Rcpp::stop("sponsor_index out of range (group %d)", gid); 
    }
      
    double mu_m = curEx(si, t_ref); 
    double mu_s = 0.0;
    
    // with a group-weighted mean of sponsor positions at their own periods:
    //double mu_m = 0.0;
    //double wsum = 0.0;
    //for (uword k = 0; k < items.size(); ++k) {
    //  uword j  = items[k];
    //  int t    = (int)bill_session(j,0);
    //  int si_j = (int)sponsor_index(j,0) - 1;
    //  if (si_j < 0 || si_j >= (int)nN) continue;
    //  // weight by S0_t (size of present set) to stabilize
    //  const mat XtXt = curEx2x2.slice((uword)t);
    //  double w = std::max(1.0, XtXt(0,0));
    //  mu_m += w * curEx(si_j, t);
    //  wsum += w;
    //}
    //if (wsum > 0) mu_m /= wsum; else mu_m = curEx(si, t_ref);
    //double mu_s = 0.0;
    
    
    // choose prior mean for m: earliest occurrence sponsor_x (default)
    //int j_ref = (int)items.front();
    //int t_ref = (int)bill_session(j_ref,0);
    //for (uword k = 1; k < items.size(); ++k) {
    //  int t = (int)bill_session(items[k],0);
    //  if (t < t_ref) { t_ref = t; j_ref = (int)items[k]; }
    //}
    //int si = (int)sponsor_index(j_ref,0) - 1; // 1-based -> 0-based
    //if (si < 0 || si >= (int)nN) Rcpp::stop("sponsor_index out of range (group %d)", gid);
    //double mu_m = curEx(si, t_ref);
    //double mu_s = 0.0;
    
    // Accumulate pooled S's
    for (uword k = 0; k < items.size(); ++k) {
      uword j = items[k];
      int   t = (int)bill_session(j,0);
      const mat XtXt = curEx2x2.slice((uword)t);
      S0  += XtXt(0,0);
      Sx  += XtXt(0,1);
      Sx2 += XtXt(1,1);
      
      for (uword i = 0; i < nN; ++i) {
        if (t < startlegis(i,0) || t > endlegis(i,0)) continue;
        double yst = curEystar(i,j);
        double pit = curEp(i,t);
        double xit = curEx(i,t);
        double r   = yst - pit;
        Sy  += r;
        Syx += r * xit;
      }
    }
    
    // start from any member's current (m,s) (doesn't matter—they’ll all be overwritten)
    double m = curEm( items[0], 0 );
    double s = curEs( items[0], 0 );
    
    arma::mat Lambda_g = ((double)items.size()) * Lambda;
    
    // reuse the exact same Newton step as your per-item solver
    arma::mat H_A(2,2); H_A(0,0) = -2.0; H_A(1,1) = 2.0; H_A(0,1)=H_A(1,0)=0.0;
    
    double prev_obj = std::numeric_limits<double>::infinity();
    for (unsigned it = 0; it < newton_maxit; ++it) {
      double A = s*s - m*m;
      double B = 2.0*(m - s);
      
      arma::vec gA(2), gB(2);
      gA(0) = -2.0*m;  gA(1) =  2.0*s;
      gB(0) =  2.0;    gB(1) = -2.0;
      
      arma::vec theta(2); theta(0)=m; theta(1)=s;
      arma::vec mu(2);    mu(0)=mu_m; mu(1)=mu_s;
      
      const double quad  = 0.5*( S0*sq(A) + 2.0*Sx*A*B + Sx2*sq(B) - 2.0*Sy*A - 2.0*Syx*B );
      arma::vec diff = theta - mu;
      const double prior = 0.5 * arma::as_scalar( diff.t() * Lambda_g * diff );
      
      // soft box barrier
      double pm_up   = softplus(m - MS_MAX);
      double pm_low  = softplus(MS_MIN - m);
      double ps_up   = softplus(s - MS_MAX);
      double ps_low  = softplus(MS_MIN - s);
      double pen_val = LAMBDA_BAR * (pm_up + pm_low + ps_up + ps_low);
      
      double obj = quad + prior + pen_val;
      
      arma::vec grad = ( S0*A - Sy  + Sx*B ) * gA + ( Sx*A - Syx + Sx2*B ) * gB + Lambda_g * diff;
      
      // barrier grad
      grad(0) += LAMBDA_BAR * ( sigmoid(m - MS_MAX) - sigmoid(MS_MIN - m) );
      grad(1) += LAMBDA_BAR * ( sigmoid(s - MS_MAX) - sigmoid(MS_MIN - s) );
      
      arma::mat H = S0*( gA*gA.t() + A*H_A ) + Sx*( B*H_A + gA*gB.t() + gB*gA.t() ) + Sx2*( gB*gB.t() ) - Sy*H_A + Lambda_g;
        
      // barrier Hessian
      H(0,0) += LAMBDA_BAR * ( sigmoid(m - MS_MAX)*(1.0-sigmoid(m - MS_MAX)) + sigmoid(MS_MIN - m)*(1.0-sigmoid(MS_MIN - m)) );
      H(1,1) += LAMBDA_BAR * ( sigmoid(s - MS_MAX)*(1.0-sigmoid(s - MS_MAX)) + sigmoid(MS_MIN - s)*(1.0-sigmoid(MS_MIN - s)) );
        
      H(0,0) += ridge; H(1,1) += ridge;
          
      if (arma::norm(grad,2) < newton_tol) break;
      
      arma::vec step = arma::solve(H, grad, arma::solve_opts::fast);
      double step_scale = 1.0, new_m = m, new_s = s, new_obj = obj;
      
      for (int ls = 0; ls < 12; ++ls) {
        const double cand_m = m - step_scale * step(0);
        const double cand_s = s - step_scale * step(1);
        
        const double A2 = cand_s*cand_s - cand_m*cand_m;
        const double B2 = 2.0*(cand_m - cand_s);
        
        const double quad2  = 0.5*( S0*sq(A2) + 2.0*Sx*A2*B2 + Sx2*sq(B2) - 2.0*Sy*A2 - 2.0*Syx*B2 );
        arma::vec th2(2); th2(0)=cand_m; th2(1)=cand_s;
        arma::vec df2 = th2 - mu;
        const double prior2 = 0.5 * arma::as_scalar( df2.t() * Lambda_g * df2 );
        
        const double pm_up2   = softplus(cand_m - MS_MAX);
        const double pm_low2  = softplus(MS_MIN - cand_m);
        const double ps_up2   = softplus(cand_s - MS_MAX);
        const double ps_low2  = softplus(MS_MIN - cand_s);
        const double pen2     = LAMBDA_BAR * (pm_up2 + pm_low2 + ps_up2 + ps_low2);
        
        const double obj2 = quad2 + prior2 + pen2;
        
        if (obj2 <= obj) { new_m = cand_m; new_s = cand_s; new_obj = obj2; break; }
        step_scale *= 0.5;
      }
      
      m = new_m; s = new_s;
      if (std::abs(prev_obj - new_obj) < newton_tol) break;
      prev_obj = new_obj;
    }
    
    // posterior covariance (same as per-item code)
    double A = s*s - m*m;
    double B = 2.0*(m - s);
    
    arma::vec gA(2), gB(2);
    gA(0) = -2.0*m;  gA(1) =  2.0*s;
    gB(0) =  2.0;    gB(1) = -2.0;
    
    H_A.zeros(); H_A(0,0) = -2.0; H_A(1,1) =  2.0;
    
    arma::mat H = S0*( gA*gA.t() + A*H_A ) + Sx*( B*H_A + gA*gB.t() + gB*gA.t() ) + Sx2*( gB*gB.t() ) - Sy*H_A + Lambda_g;
    
    // add barrier curvature near box
    auto bcurv = [&](double z){
      double s1 = sigmoid(z - MS_MAX);     // z is m or s
      double s2 = sigmoid(MS_MIN - z);
      return LAMBDA_BAR*( s1*(1.0-s1) + s2*(1.0-s2) );
    };
    H(0,0) += bcurv(m) + ridge;
    H(1,1) += bcurv(s) + ridge;
    
    arma::mat Hs = 0.5*(H+H.t());
    double jitter = ridge;
    for (int k=0; k<6 && !Hs.is_sympd(); ++k){ jitter *= 10.0; Hs.diag() += jitter; }
    //arma::mat Sigma = Hs.is_sympd() ? arma::inv_sympd(Hs) : arma::inv(Hs + 1e-8*arma::eye<mat>(2,2));
    arma::mat Sigma;
    if (Hs.is_sympd()) {
      Sigma = arma::inv_sympd(Hs);
    } else {
      Sigma = arma::inv(Hs + 1e-8 * arma::eye<arma::mat>(2,2));
    }
    
    // soft truncation on write-back
    auto mv_m = trunc_box_scalar(m, Sigma(0,0), MS_MIN, MS_MAX);
    auto mv_s = trunc_box_scalar(s, Sigma(1,1), MS_MIN, MS_MAX);
    
    // write same (m,s,Var,Cov) to every item in the group
    for (uword k = 0; k < items.size(); ++k) {
      uword j = items[k];
      curEm(j,0)  = mv_m.first;
      curEs(j,0)  = mv_s.first;
      curVm(j,0)  = std::max(mv_m.second, 1e-12);
      curVs(j,0)  = std::max(mv_s.second, 1e-12);
      curCms(j,0) = Sigma(0,1);
    }
  } // groups
  
  // ---- Fallback: update singletons exactly as in the per-item solver ----
  for (uword j = 0; j < nJ; ++j) {
    if (anchor_group((int)j) > 0) continue;  // was handled in groups
    
    const int bs = (int)std::llround(beta_sign(j,0)); // -1,0,1
    
    int t = (int)bill_session(j,0);
    if (t < 0 || t >= (int)T) continue;
    
    const mat XtXt = curEx2x2.slice((uword)t);
    double S0  = XtXt(0,0);
    double Sx  = XtXt(0,1);
    double Sx2 = XtXt(1,1);
    
    double Sy  = 0.0, Syx = 0.0;
    for (uword i = 0; i < nN; ++i) {
      if (t < startlegis(i,0) || t > endlegis(i,0)) continue;
      double r = curEystar(i,j) - curEp(i,t);
      Sy  += r;
      Syx += r * curEx(i,t);
    }
    
    int si = (int)sponsor_index(j,0) - 1;
    if (si < 0 || si >= (int)nN) Rcpp::stop("sponsor_index out of range (singleton)");
    double mu_m = curEx(si, t);
    double mu_s = 0.0;
    
    // -------------------- SINGLETONS: HARD beta sign constraint via (c,u) --------------------
    double m0 = curEm(j,0);
    double s0 = curEs(j,0);
    
    // Reparam: m=c+d, s=c-d  where d is constrained by bs
    double c = 0.5 * (m0 + s0);
    double d0 = 0.5 * (m0 - s0);          // equals beta/4 at current point
    
    double u;                              // unconstrained optimization variable
    if (bs == 0){
      u = d0;
    } else if (bs == 1){
      // want d >= 0, initialize u so softplus(u) ≈ max(d0, small)
      u = inv_softplus(std::max(d0, 1e-8));
    } else { // bs == -1
      // want d <= 0, initialize using -softplus(u) ≈ min(d0, -small)
      u = inv_softplus(std::max(-d0, 1e-8));
    }
    
    // Newton iterations in v = (c,u)
    double prev_obj = std::numeric_limits<double>::infinity();
    
    
    for (uword it = 0; it < newton_maxit; ++it) {
      
      // map (c,u) -> (m,s) with hard sign constraint
      double d, d1, d2;
      d_from_u(bs, u, d, d1, d2);
      double m = c + d;
      double s = c - d;
      
      // --- build objective/grad/H in theta=(m,s) space exactly as before (NO sign barrier) ---
      double A = s*s - m*m;     // alpha
      double B = 2.0*(m - s);   // beta
      
      arma::vec gA(2), gB(2);
      gA(0) = -2.0*m;  gA(1) =  2.0*s;
      gB(0) =  2.0;    gB(1) = -2.0;
      
      arma::mat H_A(2,2,arma::fill::zeros);
      H_A(0,0) = -2.0; H_A(1,1) = 2.0;
      
      arma::vec theta(2); theta(0)=m; theta(1)=s;
      arma::vec mu(2);    mu(0)=mu_m; mu(1)=mu_s;
      
      const double quad  = 0.5*( S0*sq(A) + 2.0*Sx*A*B + Sx2*sq(B) - 2.0*Sy*A - 2.0*Syx*B );
      arma::vec diff = theta - mu;
      const double prior = 0.5 * arma::as_scalar( diff.t() * Lambda * diff );
      
      // soft box barrier (same as you had)
      const double pm_up   = softplus(m - MS_MAX);
      const double pm_low  = softplus(MS_MIN - m);
      const double ps_up   = softplus(s - MS_MAX);
      const double ps_low  = softplus(MS_MIN - s);
      const double pen_val = LAMBDA_BAR * (pm_up + pm_low + ps_up + ps_low);
      
      const double obj = quad + prior + pen_val;
      
      // grad_theta
      arma::vec grad_theta =
        ( S0*A - Sy  + Sx*B ) * gA
      + ( Sx*A - Syx + Sx2*B ) * gB
      + Lambda * diff;
      
      grad_theta(0) += LAMBDA_BAR * ( sigmoid(m - MS_MAX) - sigmoid(MS_MIN - m) );
      grad_theta(1) += LAMBDA_BAR * ( sigmoid(s - MS_MAX) - sigmoid(MS_MIN - s) );
      
      // Hess_theta
      arma::mat H_theta =
        S0*( gA*gA.t() + A*H_A )
        + Sx*( B*H_A + gA*gB.t() + gB*gA.t() )
        + Sx2*( gB*gB.t() )
        - Sy*H_A
        + Lambda;
        
        H_theta(0,0) += LAMBDA_BAR * ( sigmoid(m - MS_MAX)*(1.0 - sigmoid(m - MS_MAX))
                                         + sigmoid(MS_MIN - m)*(1.0 - sigmoid(MS_MIN - m)) );
        H_theta(1,1) += LAMBDA_BAR * ( sigmoid(s - MS_MAX)*(1.0 - sigmoid(s - MS_MAX))
                                         + sigmoid(MS_MIN - s)*(1.0 - sigmoid(MS_MIN - s)) );
        
        // --- transform to v=(c,u) space ---
        // J = d(m,s)/d(c,u) = [[1, d1],[1, -d1]]
        const double gm = grad_theta(0);
        const double gs = grad_theta(1);
        
        arma::vec grad_v(2);
        grad_v(0) = gm + gs;             // dc
        grad_v(1) = d1 * (gm - gs);      // du
        
        // H_v = J' H_theta J + extra from nonlinearity in u: d2*(gm - gs) added to H_uu
        const double hmm = H_theta(0,0);
        const double hms = H_theta(0,1);
        const double hss = H_theta(1,1);
        
        arma::mat H_v(2,2,arma::fill::zeros);
        H_v(0,0) = hmm + 2.0*hms + hss;                        // [1,1] H [1,1]
        H_v(0,1) = d1 * (hmm - hss);                            // [1,1] H [d1,-d1]
        H_v(1,0) = H_v(0,1);
        H_v(1,1) = d1*d1 * (hmm - 2.0*hms + hss) + d2*(gm - gs); // [d1,-d1] H [d1,-d1] + nonlin
        
        // ridge
        H_v(0,0) += ridge;
        H_v(1,1) += ridge;
        
        // convergence
        if (arma::norm(grad_v, 2) < newton_tol) {
          prev_obj = obj;
          break;
        }
        
        // Newton step in v-space
        arma::vec step(2);
        bool ok = arma::solve(step, H_v, grad_v, arma::solve_opts::fast);
        if (!ok || !step.is_finite()){
          // add jitter and try once
          H_v(0,0) += 10.0*ridge;
          H_v(1,1) += 10.0*ridge;
          ok = arma::solve(step, H_v, grad_v, arma::solve_opts::fast);
          if (!ok || !step.is_finite()) break;
        }
        
        double step_scale = 1.0;
        double new_c = c, new_u = u;
        double new_obj = obj;
        
        // backtracking line search
        for (int ls = 0; ls < 12; ++ls) {
          
          const double cand_c = c - step_scale * step(0);
          const double cand_u = u - step_scale * step(1);
          
          double cd, cd1, cd2;
          d_from_u(bs, cand_u, cd, cd1, cd2);
          const double cand_m = cand_c + cd;
          const double cand_s = cand_c - cd;
          
          const double A2 = cand_s*cand_s - cand_m*cand_m;
          const double B2 = 2.0*(cand_m - cand_s);
          
          const double quad2  = 0.5*( S0*sq(A2) + 2.0*Sx*A2*B2 + Sx2*sq(B2) - 2.0*Sy*A2 - 2.0*Syx*B2 );
          
          arma::vec th2(2); th2(0)=cand_m; th2(1)=cand_s;
          arma::vec df2 = th2 - mu;
          const double prior2 = 0.5 * arma::as_scalar( df2.t() * Lambda * df2 );
          
          const double pm_up2   = softplus(cand_m - MS_MAX);
          const double pm_low2  = softplus(MS_MIN - cand_m);
          const double ps_up2   = softplus(cand_s - MS_MAX);
          const double ps_low2  = softplus(MS_MIN - cand_s);
          const double pen2     = LAMBDA_BAR * (pm_up2 + pm_low2 + ps_up2 + ps_low2);
          
          const double obj2 = quad2 + prior2 + pen2;
          
          if (obj2 <= obj) { new_c = cand_c; new_u = cand_u; new_obj = obj2; break; }
          step_scale *= 0.5;
        }
        
        c = new_c;
        u = new_u;
        
        if (std::abs(prev_obj - new_obj) < newton_tol) break;
        prev_obj = new_obj;
    }
    
    // ---- Final map to (m,s), plus HARD projection to the box that PRESERVES the sign constraint ----
    double d, d1, d2;
    d_from_u(bs, u, d, d1, d2);
    
    double da = std::fabs(d);
    const double dmax = 0.5*(MS_MAX - MS_MIN) - 1e-8;
    if (da > dmax) {
      d = (d >= 0.0 ? 1.0 : -1.0) * dmax;
      da = dmax;
    }
    
    // Enforce MS_MIN <= c±d <= MS_MAX by clamping c
    double cmin = MS_MIN + da;
    double cmax = MS_MAX - da;
    if (c < cmin) c = cmin;
    if (c > cmax) c = cmax;
    
    double m = c + d;
    double s = c - d;
    
    // ---- Approx posterior covariance: invert Hessian in (c,u), then delta-method to (m,s) ----
    // Recompute grad_theta and H_theta at final point, then transform to H_v as above
    {
      double A = s*s - m*m;
      double B = 2.0*(m - s);
      
      arma::vec gA(2), gB(2);
      gA(0) = -2.0*m;  gA(1) =  2.0*s;
      gB(0) =  2.0;    gB(1) = -2.0;
      
      arma::mat H_A(2,2,arma::fill::zeros);
      H_A(0,0) = -2.0; H_A(1,1) = 2.0;
      
      arma::vec theta(2); theta(0)=m; theta(1)=s;
      arma::vec mu(2);    mu(0)=mu_m; mu(1)=mu_s;
      
      arma::vec diff = theta - mu;
      
      arma::vec grad_theta =
        ( S0*A - Sy  + Sx*B ) * gA
      + ( Sx*A - Syx + Sx2*B ) * gB
      + Lambda * diff;
      
      grad_theta(0) += LAMBDA_BAR * ( sigmoid(m - MS_MAX) - sigmoid(MS_MIN - m) );
      grad_theta(1) += LAMBDA_BAR * ( sigmoid(s - MS_MAX) - sigmoid(MS_MIN - s) );
      
      arma::mat H_theta =
        S0*( gA*gA.t() + A*H_A )
        + Sx*( B*H_A + gA*gB.t() + gB*gA.t() )
        + Sx2*( gB*gB.t() )
        - Sy*H_A
        + Lambda;
        
      H_theta(0,0) += LAMBDA_BAR * ( sigmoid(m - MS_MAX)*(1.0 - sigmoid(m - MS_MAX))
                                       + sigmoid(MS_MIN - m)*(1.0 - sigmoid(MS_MIN - m)) );
      H_theta(1,1) += LAMBDA_BAR * ( sigmoid(s - MS_MAX)*(1.0 - sigmoid(s - MS_MAX))
                                       + sigmoid(MS_MIN - s)*(1.0 - sigmoid(MS_MIN - s)) );
      
      const double gm = grad_theta(0);
      const double gs = grad_theta(1);
      
      const double hmm = H_theta(0,0);
      const double hms = H_theta(0,1);
      const double hss = H_theta(1,1);
      
      arma::mat H_v(2,2,arma::fill::zeros);
      H_v(0,0) = hmm + 2.0*hms + hss;
      H_v(0,1) = d1 * (hmm - hss);
      H_v(1,0) = H_v(0,1);
      H_v(1,1) = d1*d1 * (hmm - 2.0*hms + hss) + d2*(gm - gs);
      
      arma::mat Hs = 0.5*(H_v + H_v.t());
      double jitter = ridge;
      for (int k=0; k<6 && !Hs.is_sympd(); ++k){ jitter *= 10.0; Hs.diag() += jitter; }
      
      arma::mat Sigma_v;
      if (Hs.is_sympd()) Sigma_v = arma::inv_sympd(Hs);
      else               Sigma_v = arma::inv(Hs + 1e-8 * arma::eye<arma::mat>(2,2));
      
      // Delta-method to theta-space: Sigma_theta = J Sigma_v J'
      arma::mat J(2,2);
      J(0,0) = 1.0;  J(0,1) = d1;   // dm/dc, dm/du
      J(1,0) = 1.0;  J(1,1) = -d1;  // ds/dc, ds/du
      
      arma::mat Sigma_theta = J * Sigma_v * J.t();
      
      curEm(j,0)  = m;
      curEs(j,0)  = s;
      curVm(j,0)  = std::max(Sigma_theta(0,0), 1e-12);
      curVs(j,0)  = std::max(Sigma_theta(1,1), 1e-12);
      curCms(j,0) = Sigma_theta(0,1);
    }
  }
}
    
    
    
    
    