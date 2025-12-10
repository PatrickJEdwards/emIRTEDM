#include "getMS_dynIRT.h"
#include <limits>
#include <cmath>

using arma::mat; using arma::vec; using arma::uword; using arma::cube;

static inline double sq(double x){ return x*x; }

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
  // Prior precision (common to all items)
  if (!item_sigma.is_sympd()) {
    Rcpp::stop("item_sigma not SPD");
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
    int si = static_cast<int>(sponsor_index(j,0)) - 1;  // <- if R-style indices
    // If sponsor_index is 1-based, uncomment: si -= 1;
    if (si < 0 || si >= static_cast<int>(nN)) {
      Rcpp::stop("sponsor_index(%u)=%d out of range [0,%u)", j, si, nN);
    }
    double mu_m = curEx(si, t);
    double mu_s = 0.0;
    
    // Start at current means
    double m = curEm(j,0);
    double s = curEs(j,0);
    
    // Newton iterations
    double prev_obj = std::numeric_limits<double>::infinity();
    for (uword it = 0; it < newton_maxit; ++it) {
      // Transformations
      double A = s*s - m*m;     // alpha
      double B = 2.0*(m - s);   // beta
      
      // Gradient of A,B
      arma::vec gA(2), gB(2);
      gA(0) = -2.0*m;  gA(1) =  2.0*s;  // dA/dm, dA/ds
      gB(0) =  2.0;    gB(1) = -2.0;    // dB/dm, dB/ds
      
      // Hessian of A (constant), B (zero)
      arma::mat H_A(2,2,arma::fill::zeros);
      H_A(0,0) = -2.0; H_A(1,1) = 2.0;
      // H_B = 0
      
      // Objective (up to const): 0.5[ S0 A^2 + 2 Sx A B + Sx2 B^2 - 2 Sy A - 2 Syx B ] + 0.5 (theta - mu)' Λ (theta - mu)
      arma::vec theta(2); theta(0)=m; theta(1)=s;
      arma::vec mu(2);    mu(0)=mu_m; mu(1)=mu_s;
      
      double quad = 0.5*( S0*sq(A) + 2.0*Sx*A*B + Sx2*sq(B) - 2.0*Sy*A - 2.0*Syx*B );
      arma::vec diff = theta - mu;
      double prior = 0.5 * arma::as_scalar( diff.t() * Lambda * diff );
      double obj = quad + prior;
      
      // Gradient
      arma::vec grad = 
        ( S0*A - Sy + Sx*B ) * gA
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
        
        // Check convergence on gradient
        if (arma::norm(grad, 2) < newton_tol) break;
        
        // Newton step with simple damping
        arma::vec step = arma::solve(H, grad, arma::solve_opts::fast);
        double step_scale = 1.0;
        double new_m, new_s, new_obj;
        
        for (int ls = 0; ls < 10; ++ls) {
          new_m = m - step_scale * step(0);
          new_s = s - step_scale * step(1);
          
          double A2 = new_s*new_s - new_m*new_m;
          double B2 = 2.0*(new_m - new_s);
          double quad2 = 0.5*( S0*sq(A2) + 2.0*Sx*A2*B2 + Sx2*sq(B2) - 2.0*Sy*A2 - 2.0*Syx*B2 );
          arma::vec th2(2); th2(0)=new_m; th2(1)=new_s;
          arma::vec df2 = th2 - mu;
          double prior2 = 0.5 * arma::as_scalar( df2.t() * Lambda * df2 );
          new_obj = quad2 + prior2;
          
          if (new_obj <= obj) break;  // accept
          step_scale *= 0.5;          // backtrack
        }
        
        m = new_m; s = new_s;
        if (std::abs(prev_obj - new_obj) < newton_tol) break;
        prev_obj = new_obj;
    } // Newton
    
    // Final Hessian at (m,s) for posterior covariance
    {
      double A = s*s - m*m;
      double B = 2.0*(m - s);
      arma::vec gA(2), gB(2);
      gA(0) = -2.0*m;  gA(1) =  2.0*s;
      gB(0) =  2.0;    gB(1) = -2.0;
      arma::mat H_A(2,2,arma::fill::zeros);
      H_A(0,0) = -2.0; H_A(1,1) = 2.0;
      
      arma::mat H =
        S0*( gA*gA.t() + A*H_A )
        + Sx*( B*H_A + gA*gB.t() + gB*gA.t() )
        + Sx2*( gB*gB.t() )
        - Sy*H_A
        + Lambda;
        H(0,0) += ridge; H(1,1) += ridge;
        
        arma::mat Hs = 0.5 * (H + H.t());               // symmetrize
        Hs(0,0) += ridge; Hs(1,1) += ridge;
        
        // If still not SPD, jitter until it is
        double jitter = ridge;
        for (int k = 0; k < 6 && !Hs.is_sympd(); ++k) {
          jitter *= 10.0;
          Hs.diag() += jitter;
        }
        
        arma::mat Sigma;
        if (!Hs.is_sympd()) {
          Rcpp::Rcout << "H not SPD at item j=" << j << " (t=" << t << ")\n";
        }
        if (Hs.is_sympd()) {
          Sigma = arma::inv_sympd(Hs);
        } else {
          // last-resort fallback: stable but not SPD-specific
          Sigma = arma::inv(Hs + 1e-8 * arma::eye<arma::mat>(2,2));
        }
        
        
        // Write back means and (co)variances
        curEm(j,0)  = m;
        curEs(j,0)  = s;
        curVm(j,0)  = Sigma(0,0);
        curVs(j,0)  = Sigma(1,1);
        curCms(j,0) = Sigma(0,1);
    }
  } // j
}
