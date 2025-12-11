// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

#include <RcppArmadillo.h>

using namespace Rcpp;

// Build a stable joint mask (union of non-zero entries) and return aligned vectors
static inline void masked_pair_union_nonzero(const arma::mat& A,
                                             const arma::mat& B,
                                             arma::vec& A_sel,
                                             arma::vec& B_sel) {
  arma::vec av = arma::vectorise(A);
  arma::vec bv = arma::vectorise(B);
  // elementwise union mask: keep positions where either av or bv is non-zero
  arma::uvec idx = arma::find( arma::abs(av) + arma::abs(bv) > 0.0 );
  A_sel = av.elem(idx);
  B_sel = bv.elem(idx);
}

// 1 - correlation, robust to empty or constant vectors
static inline double one_minus_corr(const arma::vec& u, const arma::vec& v) {
  if (u.n_elem == 0 || v.n_elem == 0) return 0.0; // nothing to compare; don't block convergence
  double su = arma::stddev(u);
  double sv = arma::stddev(v);
  if (!std::isfinite(su) || !std::isfinite(sv) || su == 0.0 || sv == 0.0) return 0.0;
  double c = arma::as_scalar( arma::cor(arma::mat(u), arma::mat(v)) ); // 1x1
  if (!std::isfinite(c)) return 0.0;
  // clamp numerical jitter
  if (c >  1.0) c =  1.0;
  if (c < -1.0) c = -1.0;
  return 1.0 - c;
}




Rcpp::List checkConv_dynIRT(const arma::mat &oldEx,
                            const arma::mat &curEx,
                            const arma::mat &oldEb,
                            const arma::mat &curEb,
                            const arma::mat &oldEa,
                            const arma::mat &curEa,
                            const arma::mat &oldEp,   // propensities (previous)
                            const arma::mat &curEp,   // propensities (current)
                            double thresh,
                            int convtype) {
  
  double devEx = 100.0;
  double devEa = 100.0;
  double devEb = 100.0;
  double devEp = 100.0;

  // ===== Stable joint masks for Ex and Ep (union of non-zeros) =====
  arma::vec ex_old_aligned, ex_cur_aligned;
  masked_pair_union_nonzero(oldEx, curEx, ex_old_aligned, ex_cur_aligned);
  
  arma::vec ep_old_aligned, ep_cur_aligned;
  masked_pair_union_nonzero(oldEp, curEp, ep_old_aligned, ep_cur_aligned);
  
  // Detect empty after masking (warn once, but don't block convergence)
  static bool warned_ep_empty = false;
  if (ep_old_aligned.n_elem == 0 || ep_cur_aligned.n_elem == 0) {
    if (!warned_ep_empty) {
      Rcpp::warning("checkConv_dynIRT: Ep mask selected zero elements (all out-of-service or zero). "
                      "Proceeding with devEp = 0.0. Verify service-window handling upstream.");
      warned_ep_empty = true;
    }
    devEp = 0.0;
  }
  
  // For Ea/Eb (J x 1), no masking needed; compare entire vectors
  arma::vec eb_old = arma::vectorise(oldEb);
  arma::vec eb_cur = arma::vectorise(curEb);
  arma::vec ea_old = arma::vectorise(oldEa);
  arma::vec ea_cur = arma::vectorise(curEa);
  
  if (convtype == 1) {
    // correlation distance (1 - corr)
    devEx = one_minus_corr(ex_old_aligned, ex_cur_aligned);
    devEb = one_minus_corr(eb_old, eb_cur);
    devEa = one_minus_corr(ea_old, ea_cur);
    if (ep_old_aligned.n_elem > 0 && ep_cur_aligned.n_elem > 0) {
      devEp = one_minus_corr(ep_old_aligned, ep_cur_aligned);
    } else {
      devEp = 0.0;
    }
  } else if (convtype == 2) {
    // maximum absolute deviation
    devEx = (ex_cur_aligned.n_elem == ex_old_aligned.n_elem && ex_old_aligned.n_elem > 0)
    ? arma::abs(ex_cur_aligned - ex_old_aligned).max()
      : 0.0;
    devEb = arma::abs(eb_cur - eb_old).max();
    devEa = arma::abs(ea_cur - ea_old).max();
    devEp = (ep_cur_aligned.n_elem == ep_old_aligned.n_elem && ep_old_aligned.n_elem > 0)
      ? arma::abs(ep_cur_aligned - ep_old_aligned).max()
        : 0.0;
  } else {
    // Fallback: treat as correlation mode
    devEx = one_minus_corr(ex_old_aligned, ex_cur_aligned);
    devEb = one_minus_corr(eb_old, eb_cur);
    devEa = one_minus_corr(ea_old, ea_cur);
    devEp = (ep_old_aligned.n_elem > 0 && ep_cur_aligned.n_elem > 0)
      ? one_minus_corr(ep_old_aligned, ep_cur_aligned)
        : 0.0;
  }
  
  bool check = (devEx < thresh) & (devEb < thresh) & (devEa < thresh) & (devEp < thresh);
  
  return Rcpp::List::create(
    _["devEx"] = devEx,
    _["devEb"] = devEb,
    _["devEa"] = devEa,
    _["devEp"] = devEp,
    _["check"] = check
  );
}

