#include "getMS_dynIRT.h"
#include <limits>
#include <cmath>
#include <algorithm>

using arma::mat; using arma::vec; using arma::uword; using arma::cube;

static inline double sq(double x){ return x*x; }

// ---- Smooth box barrier helpers ----
static inline double softplus(double z){ return std::log1p(std::exp(z)); }
static inline double sigmoid (double z){ return 1.0 / (1.0 + std::exp(-z)); }

// ---- Stable log(exp(b) - exp(a)) with double return ----
static inline double log_diff_exp(double log_b, double log_a){
  if (log_b < log_a) std::swap(log_b, log_a);
  const double x = std::exp(log_a - log_b);
  if (x >= 1.0) return -std::numeric_limits<double>::infinity();
  return log_b + std::log1p(-x);
}

/// ---- Univariate truncated-Normal moments (for soft truncation at write-back) ----
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
  const double MS_MIN = -500.0;
  const double MS_MAX =  500.0;
  
  // Smooth barrier strength (tune 1e-3 .. 1e-1; larger = stronger push back)
  const double LAMBDA_BAR = 1e-2;
  

  // Prior precision (common to all items)
  if (!item_sigma.is_sympd()) {
    Rcpp::stop("item_sigma must be SPD");
  }
  mat Lambda = arma::inv_sympd(item_sigma); // 2x2
  
  for (uword j = 0; j < nJ; ++j) {
    
    int t = static_cast<int>(bill_session(j,0));
    if (t < 0 || t >= static_cast<int>(T)) continue;
    
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
      double r   = yst - pit;
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

    // Start from current (unconstrained)
    double m = curEm(j,0);
    double s = curEs(j,0);
    
    // Predeclare A's Hessian (reused later to avoid redeclaration)
    arma::mat H_A(2,2,arma::fill::zeros);
    H_A(0,0) = -2.0; H_A(1,1) =  2.0;
    
    // Newton iterations (projected line search)
    double prev_obj = std::numeric_limits<double>::infinity();
    for (uword it = 0; it < newton_maxit; ++it) {
      // A,B at current (m,s)
      double A = s*s - m*m;     // alpha
      double B = 2.0*(m - s);   // beta
      
      // Gradient of A,B
      arma::vec gA(2), gB(2);
      gA(0) = -2.0*m;  gA(1) =  2.0*s;  // dA/dm, dA/ds
      gB(0) =  2.0;    gB(1) = -2.0;    // dB/dm, dB/ds
      
      // Objective (up to const)
      arma::vec theta(2); theta(0)=m; theta(1)=s;
      arma::vec mu(2);    mu(0)=mu_m; mu(1)=mu_s;
      
      const double quad  = 0.5*( S0*sq(A) + 2.0*Sx*A*B + Sx2*sq(B) - 2.0*Sy*A - 2.0*Syx*B );
      arma::vec diff = theta - mu;
      const double prior = 0.5 * arma::as_scalar( diff.t() * Lambda * diff );
      
      // Smooth box barrier (softplus outside [MS_MIN, MS_MAX])
      // Penalty on each dimension: sp(m - MS_MAX) + sp(MS_MIN - m) + same for s
      double pm_up   = softplus(m - MS_MAX);
      double pm_low  = softplus(MS_MIN - m);
      double ps_up   = softplus(s - MS_MAX);
      double ps_low  = softplus(MS_MIN - s);
      double pen_val = LAMBDA_BAR * (pm_up + pm_low + ps_up + ps_low);
      
      // Barrier gradients
      double dpm_up  = LAMBDA_BAR * sigmoid(m - MS_MAX);     // d/dm sp(m-MAX)
      double dpm_low = -LAMBDA_BAR * sigmoid(MS_MIN - m);    // d/dm sp(MIN-m)
      double dps_up  = LAMBDA_BAR * sigmoid(s - MS_MAX);     // d/ds sp(s-MAX)
      double dps_low = -LAMBDA_BAR * sigmoid(MS_MIN - s);    // d/ds sp(MIN-s)
      
      // Barrier Hessian (diagonal terms only; cross-terms 0)
      double ddpm_up  = LAMBDA_BAR * sigmoid(m - MS_MAX) * (1.0 - sigmoid(m - MS_MAX));
      double ddpm_low = LAMBDA_BAR * sigmoid(MS_MIN - m) * (1.0 - sigmoid(MS_MIN - m));
      double ddps_up  = LAMBDA_BAR * sigmoid(s - MS_MAX) * (1.0 - sigmoid(s - MS_MAX));
      double ddps_low = LAMBDA_BAR * sigmoid(MS_MIN - s) * (1.0 - sigmoid(MS_MIN - s));
      
      double obj = quad + prior + pen_val;
      
      
      
      
      
      // Main gradient (model) + prior
      arma::vec grad =
        ( S0*A - Sy  + Sx*B ) * gA
      + ( Sx*A - Syx + Sx2*B ) * gB
      + Lambda * diff;
      
      
      
      // Add barrier gradient
      grad(0) += (dpm_up + dpm_low);
      grad(1) += (dps_up + dps_low);
      
      arma::mat H =
        S0*( gA*gA.t() + A*H_A )
        + Sx*( B*H_A + gA*gB.t() + gB*gA.t() )
        + Sx2*( gB*gB.t() )
        - Sy*H_A
        + Lambda;
      
      // Add barrier Hessian
      H(0,0) += (ddpm_up + ddpm_low);
      H(1,1) += (ddps_up + ddps_low);
      
      // Numerical safety: ridge if needed
      H(0,0) += ridge; H(1,1) += ridge;
      
      // Convergence on gradient norm
      if (arma::norm(grad, 2) < newton_tol) break;
      
      
      
      
      // Newton step + backtracking with projection to box
      arma::vec step = arma::solve(H, grad, arma::solve_opts::fast);
      double step_scale = 1.0;
      double new_m = m, new_s = s, new_obj = obj;
      
      // Backtracking line search (no hard projection)
      for (int ls = 0; ls < 12; ++ls) {
        const double cand_m = m - step_scale * step(0);
        const double cand_s = s - step_scale * step(1);
        
        const double A2 = cand_s*cand_s - cand_m*cand_m;
        const double B2 = 2.0*(cand_m - cand_s);
        
        const double quad2  = 0.5*( S0*sq(A2) + 2.0*Sx*A2*B2 + Sx2*sq(B2) - 2.0*Sy*A2 - 2.0*Syx*B2 );
        arma::vec th2(2); th2(0)=cand_m; th2(1)=cand_s;
        arma::vec df2 = th2 - mu;
        const double prior2 = 0.5 * arma::as_scalar( df2.t() * Lambda * df2 );
        
        // barrier at candidate
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
    } // Newton
    
    // Final Hessian at (m,s) for posterior covariance (interior formula; we’ll keep SPD + tiny floor at boundary)
    const double A = s*s - m*m;
    const double B = 2.0*(m - s);
    
    arma::vec gA(2), gB(2);
    gA(0) = -2.0*m;  gA(1) =  2.0*s;
    gB(0) =  2.0;    gB(1) = -2.0;
    
    // Reuse H_A (already defined)
    H_A(0,0) = -2.0; H_A(1,1) =  2.0;
    
    arma::mat H =
      S0*( gA*gA.t() + A*H_A )
      + Sx*( B*H_A + gA*gB.t() + gB*gA.t() )
      + Sx2*( gB*gB.t() )
      - Sy*H_A
      + Lambda;
      
    // Add barrier curvature at solution (keeps Σ stable near box)
    const double ddpm_up  = LAMBDA_BAR * sigmoid(m - MS_MAX) * (1.0 - sigmoid(m - MS_MAX));
    const double ddpm_low = LAMBDA_BAR * sigmoid(MS_MIN - m) * (1.0 - sigmoid(MS_MIN - m));
    const double ddps_up  = LAMBDA_BAR * sigmoid(s - MS_MAX) * (1.0 - sigmoid(s - MS_MAX));
    const double ddps_low = LAMBDA_BAR * sigmoid(MS_MIN - s) * (1.0 - sigmoid(MS_MIN - s));
    H(0,0) += ddpm_up + ddpm_low + ridge;
    H(1,1) += ddps_up + ddps_low + ridge;
      
      
    arma::mat Hs = 0.5 * (H + H.t()); // symmetrize
    double jitter = ridge;
    
    // SPD guarantee with jitter if needed
    for (int k=0; k<6 && !Hs.is_sympd(); ++k){
      jitter *= 10.0;
      Hs.diag() += jitter;
    }
    
    
    arma::mat Sigma;
    if (Hs.is_sympd()) Sigma = arma::inv_sympd(Hs);
    else               Sigma = arma::inv(Hs + 1e-8 * arma::eye<arma::mat>(2,2));
    

    // **Soft truncation at write-back (marginals)**
    auto mv_m = trunc_box_scalar(m, Sigma(0,0), MS_MIN, MS_MAX);
    auto mv_s = trunc_box_scalar(s, Sigma(1,1), MS_MIN, MS_MAX);

    // Write back means and (co)variances
    curEm(j,0)  = mv_m.first;
    curEs(j,0)  = mv_s.first;
    
    // Keep covariance SPD; ensure marginal floors if near boundary
    curVm(j,0)  = std::max(mv_m.second, 1e-12);
    curVs(j,0)  = std::max(mv_s.second, 1e-12);
    curCms(j,0) = Sigma(0,1);  // keep cross-cov as-is (approximation)
  } // j
}
