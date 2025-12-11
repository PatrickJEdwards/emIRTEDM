#include "getMS_dynIRT.h"
#include <limits>
#include <cmath>
#include <algorithm>

using arma::mat; using arma::vec; using arma::uword; using arma::cube;

static inline double sq(double x){ return x*x; }
static inline double clamp(double v, double lo, double hi){
  return std::min(hi, std::max(lo, v));
}

void getMS_dynIRT(mat& curEm,
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
                  unsigned int newton_maxit,
                  double newton_tol,
                  double ridge)
{
  // Hard box for item primitives
  const double MS_MIN = -10.0;
  const double MS_MAX =  10.0;
  const double VAR_FLOOR_AT_BOUND = 1e-10;

  // Prior precision (common to all items)
  if (!item_sigma.is_sympd()) {
    Rcpp::stop("item_sigma must be symmetric positive definite");
  }
  mat Lambda = arma::inv_sympd(item_sigma); // 2x2
  
  for (uword j = 0; j < nJ; ++j) {
    // Period for this item
    int t = static_cast<int>(bill_session(j,0));
    if (t < 0 || t >= static_cast<int>(T)) continue;
    
    // Shared period stats: S0, Sx, Sx2
    const mat XtXt = curEx2x2.slice((uword)t);
    double S0  = XtXt(0,0);
    double Sx  = XtXt(0,1); // == XtXt(1,0)
    double Sx2 = XtXt(1,1);
    
    // Item-specific sums over legislators present at time t
    double Sy  = 0.0;
    double Syx = 0.0;
    for (uword i = 0; i < nN; ++i) {
      if (t < startlegis(i,0) || t > endlegis(i,0)) continue;
      double yst = curEystar(i,j);
      double pit = curEp(i,t);
      double xit = curEx(i,t);
      double r   = yst - pit;   // "de-propensitied" latent utility
      Sy  += r;
      Syx += r * xit;
    }
    
    // Prior mean for m is sponsor's x at session t; s has mean 0
    int si = static_cast<int>(sponsor_index(j,0)) - 1;  // 1-based -> 0-based
    if (si < 0 || si >= static_cast<int>(nN)) {
      Rcpp::stop("sponsor_index(%u) out of range after 1-based -> 0-based shift", j);
    }
    double mu_m = curEx(si, t);
    double mu_s = 0.0;

    // Start at current means (project into box in case starts drifted out)
    double m = clamp(curEm(j,0), MS_MIN, MS_MAX);
    double s = clamp(curEs(j,0), MS_MIN, MS_MAX);
    
    // Newton iterations (projected line search)
    double prev_obj = std::numeric_limits<double>::infinity();
    for (uword it = 0; it < newton_maxit; ++it) {
      // Transformations at current (m,s)
      double A = s*s - m*m;     // alpha
      double B = 2.0*(m - s);   // beta
      
      // Gradient of A,B
      arma::vec gA(2), gB(2);
      gA(0) = -2.0*m;  gA(1) =  2.0*s;  // dA/dm, dA/ds
      gB(0) =  2.0;    gB(1) = -2.0;    // dB/dm, dB/ds
      
      // Hessian of A (constant), B (zero)
      arma::mat H_A(2,2,arma::fill::zeros);
      H_A(0,0) = -2.0; H_A(1,1) =  2.0;
      
      // Objective (up to const)
      arma::vec theta(2); theta(0)=m; theta(1)=s;
      arma::vec mu(2);    mu(0)=mu_m; mu(1)=mu_s;
      double quad = 0.5*( S0*sq(A) + 2.0*Sx*A*B + Sx2*sq(B) - 2.0*Sy*A - 2.0*Syx*B );
      arma::vec diff = theta - mu;
      double prior = 0.5 * arma::as_scalar( diff.t() * Lambda * diff );
      double obj = quad + prior;
      
      // Gradient
      arma::vec grad = 
          ( S0*A - Sy  + Sx*B ) * gA
        + ( Sx*A - Syx + Sx2*B ) * gB
        + Lambda * diff;
      
      // Hessian
      arma::mat H =
          S0*( gA*gA.t() + A*H_A )
        + Sx*( B*H_A + gA*gB.t() + gB*gA.t() )
        + Sx2*( gB*gB.t() )
        - Sy*H_A
        + Lambda;
      
      // Numerical safety: ridge if needed
      H(0,0) += ridge; H(1,1) += ridge;
      
      // Convergence on gradient norm
      if (arma::norm(grad, 2) < newton_tol) break;
      
      // Newton step + backtracking with projection to box
      arma::vec step = arma::solve(H, grad, arma::solve_opts::fast);
      double step_scale = 1.0;
      double new_m = m, new_s = s, new_obj = obj;
      
      for (int ls = 0; ls < 10; ++ls) {
        double cand_m = m - step_scale * step(0);
        double cand_s = s - step_scale * step(1);
        // Project candidate into the hard box
        cand_m = clamp(cand_m, MS_MIN, MS_MAX);
        cand_s = clamp(cand_s, MS_MIN, MS_MAX);
        
        double A2 = cand_s*cand_s - cand_m*cand_m;
        double B2 = 2.0*(cand_m - cand_s);
        double quad2 = 0.5*( S0*sq(A2) + 2.0*Sx*A2*B2 + Sx2*sq(B2) - 2.0*Sy*A2 - 2.0*Syx*B2 );
        arma::vec th2(2); th2(0)=cand_m; th2(1)=cand_s;
        arma::vec df2 = th2 - mu;
        double prior2 = 0.5 * arma::as_scalar( df2.t() * Lambda * df2 );
        double obj2 = quad2 + prior2;
        
        if (obj2 <= obj) { new_m = cand_m; new_s = cand_s; new_obj = obj2; break; }
        step_scale *= 0.5;
      }
      
      m = new_m; s = new_s;
      if (std::abs(prev_obj - new_obj) < newton_tol) break;
      prev_obj = new_obj;
    } // Newton
    
    // Clamp once more post-iteration
    m = clamp(m, MS_MIN, MS_MAX);
    s = clamp(s, MS_MIN, MS_MAX);
    
    // Final Hessian at (m,s) for posterior covariance (interior formula; we’ll keep SPD + tiny floor at boundary)
    double A = s*s - m*m;
    double B = 2.0*(m - s);
    arma::vec gA(2), gB(2);
    gA(0) = -2.0*m;  gA(1) =  2.0*s;
    gB(0) =  2.0;    gB(1) = -2.0;
    arma::mat H_A(2,2,arma::fill::zeros);
    H_A(0,0) = -2.0; H_A(1,1) =  2.0;
    
    arma::mat H =
        S0*( gA*gA.t() + A*H_A )
      + Sx*( B*H_A + gA*gB.t() + gB*gA.t() )
      + Sx2*( gB*gB.t() )
      - Sy*H_A
      + Lambda;
    arma::mat Hs = 0.5 * (H + H.t());
    Hs(0,0) += ridge; Hs(1,1) += ridge;
    
    // SPD guarantee with jitter if needed
    double jitter = ridge;
    for (int k = 0; k < 6 && !Hs.is_sympd(); ++k) {
      jitter *= 10.0;
      Hs.diag() += jitter;
    }
    
    arma::mat Sigma;
    if (Hs.is_sympd()) {
      Sigma = arma::inv_sympd(Hs);
    } else {
      Sigma = arma::inv(Hs + 1e-8 * arma::eye<arma::mat>(2,2));
    }

    // If we’re stuck on a boundary, keep a tiny positive variance to avoid degeneracy downstream
    bool at_bound = (m <= MS_MIN + 1e-12) || (m >= MS_MAX - 1e-12) ||
                    (s <= MS_MIN + 1e-12) || (s >= MS_MAX - 1e-12);
    if (at_bound) {
      Sigma(0,0) = std::max(Sigma(0,0), VAR_FLOOR_AT_BOUND);
      Sigma(1,1) = std::max(Sigma(1,1), VAR_FLOOR_AT_BOUND);
    }

    // Write back means and (co)variances
    curEm(j,0)  = m;
    curEs(j,0)  = s;
    curVm(j,0)  = Sigma(0,0);
    curVs(j,0)  = Sigma(1,1);
    curCms(j,0) = Sigma(0,1);
  } // j
}
