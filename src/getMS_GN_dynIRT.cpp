// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
#include "getMS_GN_dynIRT.h"

using namespace Rcpp;

// Helpers: transform (m,s) -> (alpha, beta) and their gradients
static inline double alpha_of(double m, double s){ return s*s - m*m; }
static inline double beta_of (double m, double s){ return 2.0*(m - s); }

// ∇β = [2, -2], ∇α = [-2m, 2s]
static inline arma::vec grad_beta(double m, double s){
  arma::vec g(2); g(0)=2.0; g(1)=-2.0; return g;
}
static inline arma::vec grad_alpha(double m, double s){
  arma::vec g(2); g(0)=-2.0*m; g(1)= 2.0*s; return g;
}

// f_i(m,s;x_i) = α + β x_i = (s^2 - m^2) + 2(m - s) x_i
// Residual r_i = ydag_i - f_i
// Jacobian row J_i = [∂f/∂m, ∂f/∂s] = [-2m + 2x_i, 2s - 2x_i]
void getMS_GN_dynIRT(arma::mat &Em, arma::mat &Es, arma::cube &Vm2,
                     arma::mat &Ea, arma::mat &Eb,
                     arma::mat &Ebb, arma::mat &Eba,
                     const arma::mat &Eystar,    // N x J
                     const arma::mat &Ex,        // N x T
                     const arma::mat &Ep,        // N x T
                     const arma::mat &bill_session, // J x 1
                     const arma::mat &ones_col,      // N x T
                     const arma::mat &mu_ms,         // 2 x J  (or 2 x 1)
                     const arma::mat &Sigma_ms,      // 2 x 2
                     const unsigned int max_newton){
  
  const int N = Eystar.n_rows;
  const int J = Eystar.n_cols;
  
  arma::mat SigInv = arma::inv_sympd(Sigma_ms);
  
  // broadcast prior means if provided as 2x1
  bool broadcast_mu = (mu_ms.n_cols == 1 && mu_ms.n_rows == 2);
  
#pragma omp parallel for
  for(int j=0; j<J; ++j){
    const int t = static_cast<int>(bill_session(j,0));
    
    // Build masks + vectors for time t
    arma::vec xi  = Ex.col(t);                  // N x 1
    arma::vec mask= ones_col.col(t);            // N x 1 (0/1)
    arma::vec yj  = Eystar.col(j);              // N x 1
    arma::vec pit = Ep.col(t);                  // N x 1
    
    // y^dagger = E[y*] - E[p] (masked)
    arma::vec ydag = yj - (pit % mask);
    
    // Prior mean for (m,s)
    arma::vec mu = broadcast_mu ? mu_ms.col(0) : mu_ms.col(j);  // 2 x 1
    
    // Initialize at previous Em,Es (or prior mean if not set)
    double m = Em(j,0);
    double s = Es(j,0);
    if(!std::isfinite(m) || !std::isfinite(s)){
      m = mu(0); s = mu(1);
    }
    
    // Gauss–Newton iterations
    arma::mat H(2,2); arma::vec g(2); arma::vec step(2);
    for(unsigned int it=0; it<max_newton; ++it){
      // residuals and Jacobian accumulators (masked)
      double g0=0.0, g1=0.0;
      double h00=SigInv(0,0), h01=SigInv(0,1), h11=SigInv(1,1); // start with prior precision
      arma::vec r(N, arma::fill::zeros);
      
      for(int i=0; i<N; ++i){
        if(mask(i)==0.0) continue;  // out-of-service
        const double x = xi(i);
        const double fi = (s*s - m*m) + 2.0*(m - s)*x;   // α + β x
        const double ri = ydag(i) - fi;
        r(i) = ri;
        // J_i
        const double Jm = -2.0*m + 2.0*x;
        const double Js =  2.0*s - 2.0*x;
        
        // g += Jᵀ r
        g0 += Jm * ri;
        g1 += Js * ri;
        
        // H += Jᵀ J
        h00 += Jm*Jm;
        h01 += Jm*Js;
        h11 += Js*Js;
      }
      
      H(0,0)=h00; H(0,1)=h01;
      H(1,0)=h01; H(1,1)=h11;
      
      // add prior linear term: + SigInv (mu - theta)
      arma::vec theta(2); theta(0)=m; theta(1)=s;
      arma::vec prior_lin = SigInv * (mu - theta);
      
      g(0)=g0 + prior_lin(0);
      g(1)=g1 + prior_lin(1);
      
      // solve H step = g
      step = arma::solve(H, g, arma::solve_opts::likely_sympd);
      
      // update
      m += step(0);
      s += step(1);
      
      // tiny safeguard: if step is extremely small, stop early
      if(std::abs(step(0)) + std::abs(step(1)) < 1e-10) break;
    }
    
    // Store mode (posterior mean under Laplace)
    Em(j,0)=m; Es(j,0)=s;
    
    // Laplace covariance = H^{-1} at the mode
    Vm2.slice(j) = arma::inv_sympd(H);
    
    // Transform to (α,β) at the mode
    const double alpha = alpha_of(m,s);
    const double beta  = beta_of(m,s);
    Ea(j,0)=alpha; Eb(j,0)=beta;
    
    // Delta-method second moments
    arma::vec gb = grad_beta(m,s);
    arma::vec ga = grad_alpha(m,s);
    const double var_beta  = arma::as_scalar(gb.t() * Vm2.slice(j) * gb);
    const double var_alpha = arma::as_scalar(ga.t() * Vm2.slice(j) * ga);
    const double cov_ba    = arma::as_scalar(gb.t() * Vm2.slice(j) * ga);
    
    // E[β^2] ≈ β^2 + Var(β) ;   E[β α] ≈ β α + Cov(β,α)
    Ebb(j,0) = beta*beta + var_beta;
    Eba(j,0) = beta*alpha + cov_ba;
  }
}
