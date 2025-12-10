// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>

using namespace Rcpp;

static inline double clamp(double v, double lo, double hi){
  return std::min(hi, std::max(lo, v));
}

// // [[Rcpp::export()]]
void getX_dynIRT(arma::mat &Ex,
                 arma::mat &Vx,
                 const arma::mat &Ebb,          // J×1, element j:  E[β_{jt}^2]   (stacked by time blocks)
                 const arma::mat &omega2,       // N×1, ω_i^2
                 const arma::mat &Eb,           // J×1, element j:  E[β_{jt}]     (stacked by time blocks)
                 const arma::mat &Eystar,       // N×J, E[y*_{ijt}]
                 const arma::mat &Eba,          // J×1, element j:  E[β_{jt} α_{jt}] (stacked by time blocks)
                 const arma::mat &startlegis,   // N×1
                 const arma::mat &endlegis,     // N×1
                 const arma::mat &xmu0,         // N×1, c_{i,T_i−1} prior mean
                 const arma::mat &xsigma0,      // N×1, C_{i,T_i−1} prior var
                 const int T,
                 const int N,
                 const arma::mat &end_session,  // T×1, end indices per time block in {items j}
                 const arma::mat &Ep             // N×T, E[p_{it}]
) {


	int i, t;

  const double X_MAX = 10.0;     // upper bound for ideal points
  const double X_MIN = -10.0;    // lower bound for ideal points
  const double EPS   = 1e-8;     // small numeric floor
  const double EPSV  = 1e-12;    // min variance

  // ===== Per-time aggregates over items =====
  arma::mat betaDD( T,1,arma::fill::zeros);  // 𝛽̈_t  = sqrt( Σ_j E[β_{jt}^2] )
  arma::mat Eba_sum(T,1,arma::fill::zeros);  // Σ_j E[β_{jt} α_{jt}]
  arma::mat Eb_sum( T,1,arma::fill::zeros);  // Σ_j E[β_{jt}] (not used in current yDD)

  // ===== Per-(i,t) temporary slices =====
  arma::mat Eby_sum, Eb_t, Eystar_t;        // Eby_sum = (Eystar_t * Eb_t) = Σ_j E[y*_{ijt}] E[β_{jt}]
  
  // ===== Kalman work arrays (per i,t) =====
  arma::mat Ot(N,T,arma::fill::zeros);      // Ω_t   (prediction variance for x)
  arma::mat Kt(N,T,arma::fill::zeros);      // K_t   (Kalman gain)
  arma::mat St(N,T,arma::fill::zeros);      // S_t   (innovation variance = 𝛽̈_t^2 Ω_t + 1)
  arma::mat Jt(N,T,arma::fill::zeros);      // J_t   (smoother gain)
  arma::mat C_var(N,T,arma::fill::zeros);   // C_t   (filtered variance)
  arma::mat c_mean(N,T,arma::fill::zeros);  // c_t   (filtered mean)
	
	double yDD;                                // ẏ̈_{it}  (collapsed pseudo-observation for x at time t)
	
	
	// ---------- Build per-time aggregates over items ----------
	// betaDD and Eba_sum were already used; we add Eb_sum to capture Σ_j E[β_{jt}] so we can subtract p_{it} times this quantity.
	// These quantities are called repeatedly, calculate and store for reuse
	//betaDD and yDD correspond to beta.dot.dot and y.dot.dot respectively
	betaDD(0,0)  = std::sqrt(std::max(0.0, arma::accu(Ebb.submat(0,0, end_session(0,0)-1,0))));
  Eba_sum(0,0) = arma::accu(Eba.submat(0,0, end_session(0,0)-1,0));
  Eb_sum(0,0)  = arma::accu(Eb .submat(0,0,  end_session(0,0)-1,0));

  if(T > 1){
    #pragma omp parallel for
    for(t = 1; t < T; t++){
      betaDD(t,0)  = std::sqrt(std::max(0.0, arma::accu(Ebb.submat(end_session(t-1,0),0, end_session(t,0)-1,0))));
      Eba_sum(t,0) = arma::accu(Eba.submat(end_session(t-1,0),0, end_session(t,0)-1,0));
      Eb_sum(t,0)  = arma::accu(Eb .submat(end_session(t-1,0),0, end_session(t,0)-1,0));
    }
  }
	
	

	// ---------- Kalman forward–backward per legislator ----------
  #pragma omp parallel for private(t,Eystar_t,Eb_t,Eby_sum,yDD)
  for(i=0; i < N; i++){

    // Initialize at first served period t0
    t = startlegis(i,0);

    if(t==0){
      Eystar_t = Eystar.submat(i,0,i,end_session(t,0)-1);
      Eb_t     = Eb    .submat(0,0,end_session(t,0)-1,0);
    } else {
      Eystar_t = Eystar.submat(i,end_session(t-1,0),i,end_session(t,0)-1);
      Eb_t     = Eb    .submat(    end_session(t-1,0),0,end_session(t,0)-1,0);
    }

    // Σ_j E[y*_{ijt}] E[β_{jt}]
    arma::rowvec ydagger_t = Eystar_t - Ep(i,t);   // broadcast scalar p_it
    Eby_sum = ydagger_t * Eb_t;                    // scalar
		
		// ẏ̈_{it} = [ Σ_j E[y*]E[β]  −  Σ_j E[β α] ] / 𝛽̈_t
		double denom = std::max(betaDD(t,0), EPS);
		yDD          = ( Eby_sum(0,0) - Eba_sum(t,0) ) / denom;
		
		// ---- Kalman filter update at entry period ----
		Ot(i,t)    = omega2(i,0) + xsigma0(i,0);                 // Ω_t = ω_i^2 + C_{i0}
		St(i,t)    = betaDD(t,0)*betaDD(t,0)*Ot(i,t) + 1;        // S_t = 𝛽̈_t^2 Ω_t + 1
		Kt(i,t)    = (St(i,t)>EPS)? betaDD(t,0)*Ot(i,t)/St(i,t) : 0.0; // K_t
		C_var(i,t) = std::max((1 - Kt(i,t)*betaDD(t,0))*Ot(i,t), EPSV); // C_t
		c_mean(i,t)= xmu0(i,0) + Kt(i,t)*(yDD - betaDD(t,0)*xmu0(i,0)); // c_t (filtered)
		
		// Clamp filtered mean at entry
		c_mean(i,t) = clamp(c_mean(i,t), X_MIN, X_MAX);
		
		// If only one served period
		if(startlegis(i,0) == endlegis(i,0)){
		  Vx(i,t) = std::max(C_var(i,t), EPSV);
		  Ex(i,t) = c_mean(i,t);                // already clamped
		}

		// If legislators in multiple periods (should be most instances)
		if(startlegis(i,0) != endlegis(i,0)){
		  
		  // ---- Forward filtering over subsequent served periods ----
		  for(t = startlegis(i,0) + 1; t <= endlegis(i,0); t++){
		    
		    Eystar_t = Eystar.submat(i, end_session(t-1,0), i, end_session(t,0)-1);
		    Eb_t     = Eb    .submat(    end_session(t-1,0), 0, end_session(t,0)-1, 0);
		    
		    arma::rowvec ydagger_t2 = Eystar_t - Ep(i,t);   // broadcast p_it
		    Eby_sum = ydagger_t2 * Eb_t;                    // scalar
		    
		    denom = std::max(betaDD(t,0), EPS);
		    yDD   = ( Eby_sum(0,0) - Eba_sum(t,0) ) / denom;
		    
		    Ot(i,t)    = omega2(i,0) + C_var(i,t-1);                   // Ω_t
		    St(i,t)    = betaDD(t,0)*betaDD(t,0)*Ot(i,t) + 1;          // S_t
		    Kt(i,t)    = (St(i,t)>EPS)? betaDD(t,0)*Ot(i,t)/St(i,t) : 0.0; // K_t
		    C_var(i,t) = std::max((1 - Kt(i,t)*betaDD(t,0))*Ot(i,t), EPSV);
		    c_mean(i,t)= c_mean(i,t-1) + Kt(i,t)*(yDD - betaDD(t,0)*c_mean(i,t-1));
		    
		    // Clamp filtered mean each step
		    c_mean(i,t) = clamp(c_mean(i,t), X_MIN, X_MAX);
		  }
		  
		  // ---- Backward Rauch–Tung–Striebel smoother ----
		  Vx(i, endlegis(i,0)) = std::max(C_var(i,endlegis(i,0)), EPSV);
		  Ex(i, endlegis(i,0)) = clamp(c_mean(i,endlegis(i,0)), X_MIN, X_MAX);
		  
		  for(t = endlegis(i,0) - 1; t >= startlegis(i,0); t--){
		    Jt(i,t) = (Ot(i,t+1)>EPS)? (C_var(i,t)/Ot(i,t+1)) : 0.0;   // J_t
		    Vx(i,t) = std::max(C_var(i,t) + Jt(i,t)*Jt(i,t)*(Vx(i,t+1) - Ot(i,t+1)), EPSV);
		    Ex(i,t) = c_mean(i,t) + Jt(i,t)*(Ex(i,t+1) - c_mean(i,t));
		    
		    // Clamp smoothed mean; keep variance positive
		    Ex(i,t) = clamp(Ex(i,t), X_MIN, X_MAX);
		  }
		} // multi-period
  } 	// for i

	return;
}
