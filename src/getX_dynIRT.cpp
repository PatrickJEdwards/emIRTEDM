// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>
#include <limits>      // for std::numeric_limits<double>::infinity()
#include <algorithm>   // for std::min/max

using namespace Rcpp;

// Truncated-Normal moments on a finite box [L, U] for scalar Normal N(mu,var)
static inline std::pair<double,double>
  trunc_box(double mu, double var, double L, double U){
    const double VAR_FLOOR = 1e-12;
    var = std::max(var, VAR_FLOOR);
    if (!(U > L)) return { std::min(U,std::max(L,mu)), VAR_FLOOR };
    
    const double sd = std::sqrt(var);
    const double a  = (L - mu)/sd;
    const double b  = (U - mu)/sd;
    
    const double log_phi_a = R::dnorm(a, 0.0, 1.0, /*log*/true);
    const double log_phi_b = R::dnorm(b, 0.0, 1.0, /*log*/true);
    const double log_Phi_a = R::pnorm(a, 0.0, 1.0, /*lower*/true,  /*log*/true);
    const double log_Phi_b = R::pnorm(b, 0.0, 1.0, /*lower*/true,  /*log*/true);
    
    // log( Phi(b) - Phi(a) )
    auto log_diff_exp = [](double log_b, double log_a) -> double {
      if (log_b < log_a) std::swap(log_b, log_a);
      double x = std::exp(log_a - log_b);
      if (x >= 1.0) return -std::numeric_limits<double>::infinity();
      return log_b + std::log1p(-x);
    };
    double logZ = log_diff_exp(log_Phi_b, log_Phi_a);
    if (!std::isfinite(logZ)) return { std::min(U,std::max(L,mu)), VAR_FLOOR };
    
    const double r_a = std::exp(log_phi_a - logZ);
    const double r_b = std::exp(log_phi_b - logZ);
    
    const double mean = mu + sd * (r_a - r_b);
    const double term = (a * r_a - b * r_b);
    double variance   = var * (1.0 + term - (r_a - r_b)*(r_a - r_b));
    if (!(variance > 0.0) || !std::isfinite(variance)) variance = VAR_FLOOR;
    
    // Small safety (should be already inside)
    double mean_box = std::min(U, std::max(L, mean));
    return { mean_box, variance };
  }





// // [[Rcpp::export()]]
void getX_dynIRT(arma::mat &Ex,
                 arma::mat &Vx,
                 const arma::mat &Ebb,          // J×1: E[β^2] stacked by time
                 const arma::mat &omega2,       // N×1: process noise per i
                 const arma::mat &Eb,           // J×1: E[β] stacked by time
                 const arma::mat &Eystar,       // N×J: E[y*]
                 const arma::mat &Eba,          // J×1: E[β α] stacked by time
                 const arma::mat &startlegis,   // N×1
                 const arma::mat &endlegis,     // N×1
                 const arma::mat &xmu0,         // N×1: prior mean at entry
                 const arma::mat &xsigma0,      // N×1: prior var at entry
                 const int T,
                 const int N,
                 const arma::mat &end_session,  // T×1: end indices in {items j}
                 const arma::mat &Ep            // N×T: E[p_it]
) {


	int i, t;

  const double X_MIN =   -500.0;  // lower bound for ideal points
  const double X_MAX =    500.0;  // upper bound for ideal points
  const double EPS   =    1e-8;  // small numeric floor
  const double EPSV  =    1e-12; // min variance
  const double BETA_EPS = 1e-8;  // treat betaDD≈0 as no information

  // ===== Per-time aggregates over items =====
  arma::mat betaDD( T,1,arma::fill::zeros);  // sqrt( Σ_j E[β_{jt}^2] )
  arma::mat Eba_sum(T,1,arma::fill::zeros);  // Σ_j E[β_{jt} α_{jt}]
  arma::mat Eb_t;                            // slice E[β] per t
  arma::mat Eystar_t;
  
  
	
	// ---------- Build per-time aggregates over items ----------
	// betaDD and Eba_sum were already used; we add Eb_sum to capture Σ_j E[β_{jt}] so we can subtract p_{it} times this quantity.
	// These quantities are called repeatedly, calculate and store for reuse
	//betaDD and yDD correspond to beta.dot.dot and y.dot.dot respectively
	betaDD(0,0)  = std::sqrt(std::max(0.0, arma::accu(Ebb.submat(0,0, end_session(0,0)-1,0))));
  Eba_sum(0,0) = arma::accu(Eba.submat(0,0, end_session(0,0)-1,0));
  if (T > 1){
    #pragma omp parallel for
    for(int tt = 1; tt < T; ++tt){
      betaDD(tt,0)  = std::sqrt(std::max(0.0, arma::accu(Ebb.submat(end_session(tt-1,0),0, end_session(tt,0)-1,0))));
      Eba_sum(tt,0) = arma::accu(Eba.submat(end_session(tt-1,0),0, end_session(tt,0)-1,0));
    }
  }
	
	// ===== Kalman work arrays (per i,t) =====
	arma::mat Ot(N,T,arma::fill::zeros);      // Ω_t   (prediction variance for x)
  arma::mat Kt(N,T,arma::fill::zeros);      // K_t   (Kalman gain)
  arma::mat St(N,T,arma::fill::zeros);      // S_t   (innovation variance = 𝛽̈_t^2 Ω_t + 1)
  arma::mat Jt(N,T,arma::fill::zeros);      // J_t   (smoother gain)
  arma::mat C_var(N,T,arma::fill::zeros);   // C_t   (filtered variance)
  arma::mat c_mean(N,T,arma::fill::zeros);  // c_t   (filtered mean)
  
  double yDD;                                // ẏ̈_{it}  (collapsed pseudo-observation for x at time t)
	

	// ---------- Kalman forward–backward per legislator ----------
  #pragma omp parallel for private(t,Eystar_t,Eb_t,yDD)
	for(i=0; i < N; i++){
	  
	  int t0 = startlegis(i,0);
	  int t1 = endlegis(i,0);
	  
	  // Skip MPs that never serve
	  if (t0 < 0 || t1 < t0 || t1 >= T){
	    for (int tt=0; tt<T; ++tt){ Ex(i,tt)=0.0; Vx(i,tt)=0.0; }
	    continue;
	  }
	  
	  // ---- Entry period t0
	  t = t0;
	  if (t == 0){
	    Eystar_t = Eystar.submat(i, 0, i, end_session(t,0)-1);
	    Eb_t     = Eb     .submat(0, 0, end_session(t,0)-1, 0);
	  } else {
	    Eystar_t = Eystar.submat(i, end_session(t-1,0), i, end_session(t,0)-1);
	    Eb_t     = Eb     .submat(    end_session(t-1,0), 0, end_session(t,0)-1, 0);
	  }
	  

	  arma::rowvec ydagger_t = Eystar_t - Ep(i,t); // subtract p_it once (correct)
	  double EyEb = arma::as_scalar( ydagger_t * Eb_t ); // Σ_j E[y*−p] E[β]
	  double denom = std::max(betaDD(t,0), EPS);
	  yDD = ( EyEb - Eba_sum(t,0) ) / denom;
	  
		// ---- Kalman filter update at entry period ----
		Ot(i,t)    = omega2(i,0) + xsigma0(i,0);                 // Ω_t = ω_i^2 + C_{i0}
		
		// Propagate-only when betaDD is tiny (no measurement info)
		if (betaDD(t,0) < BETA_EPS) {
		  Kt(i,t)    = 0.0;
		  St(i,t)    = 1.0;                  // arbitrary positive; not used further
		  C_var(i,t) = std::max(Ot(i,t), EPSV);
		  c_mean(i,t)= xmu0(i,0);            // prior mean at entry
		  
		  auto mv = trunc_box(c_mean(i,t), C_var(i,t), X_MIN, X_MAX);
		  c_mean(i,t) = mv.first;
		  C_var(i,t)  = mv.second;
		} else {
		  // Measurement update
		  arma::rowvec ydagger_t = Eystar_t - Ep(i,t);          // subtract p_it once (correct)
		  double EyEb = arma::as_scalar( ydagger_t * Eb_t );    // Σ_j (E[y*−p] E[β])
		  double denom = std::max(betaDD(t,0), EPS);
		  yDD = ( EyEb - Eba_sum(t,0) ) / denom;
		  
		  St(i,t)    = betaDD(t,0)*betaDD(t,0)*Ot(i,t) + 1.0;                    // S_t
		  Kt(i,t)    = (St(i,t)>EPS)? betaDD(t,0)*Ot(i,t)/St(i,t) : 0.0;         // K_t
		  C_var(i,t) = std::max((1.0 - Kt(i,t)*betaDD(t,0))*Ot(i,t), EPSV);      // C_t
		  c_mean(i,t)= xmu0(i,0) + Kt(i,t)*(yDD - betaDD(t,0)*xmu0(i,0));        // c_t
		  
		  auto mv = trunc_box(c_mean(i,t), C_var(i,t), X_MIN, X_MAX);
		  c_mean(i,t) = mv.first;
		  C_var(i,t)  = mv.second;
		}
		
		
		// If only one served period
		if (t0 == t1){
		  Vx(i,t) = C_var(i,t);
		  Ex(i,t) = c_mean(i,t);
		  continue;
		}

		// If legislators in multiple periods (should be most instances)
		if(startlegis(i,0) != endlegis(i,0)){
		  
		  // ---- Forward filtering over subsequent served periods ----
		  for(t = t0+1; t <= t1; ++t){
		    
		    Eystar_t = Eystar.submat(i, end_session(t-1,0), i, end_session(t,0)-1);
		    Eb_t     = Eb     .submat(    end_session(t-1,0), 0, end_session(t,0)-1, 0);
		    
		    arma::rowvec ydagger_t2 = Eystar_t - Ep(i,t);
		    EyEb  = arma::as_scalar( ydagger_t2 * Eb_t );
		    denom = std::max(betaDD(t,0), EPS);
		    yDD   = ( EyEb - Eba_sum(t,0) ) / denom;
		    
		    Ot(i,t)    = omega2(i,0) + C_var(i,t-1);
		    
		    // propagate-only when betaDD is tiny (no measurement info)
		    if (betaDD(t,0) < BETA_EPS) {
		      Kt(i,t)    = 0.0;
		      St(i,t)    = 1.0;
		      C_var(i,t) = std::max(Ot(i,t), EPSV);
		      c_mean(i,t)= c_mean(i,t-1);     // carry forward prior mean
		      
		      auto mv = trunc_box(c_mean(i,t), C_var(i,t), X_MIN, X_MAX);
		      c_mean(i,t) = mv.first;
		      C_var(i,t)  = mv.second;
		    } else {
		      // Measurement update
		      arma::rowvec ydagger_t2 = Eystar_t - Ep(i,t);
		      double EyEb  = arma::as_scalar( ydagger_t2 * Eb_t );
		      double denom = std::max(betaDD(t,0), EPS);
		      yDD   = ( EyEb - Eba_sum(t,0) ) / denom;
		      
		      St(i,t)    = betaDD(t,0)*betaDD(t,0)*Ot(i,t) + 1.0;
		      Kt(i,t)    = (St(i,t)>EPS)? betaDD(t,0)*Ot(i,t)/St(i,t) : 0.0;
		      C_var(i,t) = std::max((1.0 - Kt(i,t)*betaDD(t,0))*Ot(i,t), EPSV);
		      c_mean(i,t)= c_mean(i,t-1) + Kt(i,t)*(yDD - betaDD(t,0)*c_mean(i,t-1));
		      
		      auto mv = trunc_box(c_mean(i,t), C_var(i,t), X_MIN, X_MAX);
		      c_mean(i,t) = mv.first;
		      C_var(i,t)  = mv.second;
		    }
		  }
		  
		  // ---- Backward Rauch–Tung–Striebel smoother ----
		  Vx(i, t1) = C_var(i, t1);
		  Ex(i, t1) = c_mean(i, t1);
		  
		  {
		    auto mv = trunc_box(Ex(i,t1), Vx(i,t1), X_MIN, X_MAX);
		    Ex(i,t1) = mv.first;
		    Vx(i,t1) = mv.second;
		  }
		  
		  for(t = t1-1; t >= t0; --t){
		    Jt(i,t) = (Ot(i,t+1)>EPS)? (C_var(i,t)/Ot(i,t+1)) : 0.0;
		    Vx(i,t) = std::max(C_var(i,t) + Jt(i,t)*Jt(i,t)*(Vx(i,t+1) - Ot(i,t+1)), EPSV);
		    Ex(i,t) = c_mean(i,t) + Jt(i,t)*(Ex(i,t+1) - c_mean(i,t));
		    
		    auto mv = trunc_box(Ex(i,t), Vx(i,t), X_MIN, X_MAX);
		    Ex(i,t) = mv.first;
		    Vx(i,t) = mv.second;
		  }
		} // multi-period
  } 	// for i

	return;
}
