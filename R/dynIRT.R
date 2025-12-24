dynIRT <- function(.data,
                    .starts = NULL,
                    .priors = NULL,
                    .control = NULL
                    ) {
    cl <- match.call()

    divider <- c(paste(rep("=", 20), sep = "", collapse = ""), "\n")

    ## Default Control
    default <- list(threads = 1L,
                    verbose = FALSE,
                    maxit = 500,
                    thresh = 1e-6,
                    checkfreq = 50
                    )
    cat("\n")
    cat(divider)
    cat("dynIRT: Dynamic IRT via Variational Inference\n\n")

    ## Main Call to Computation
    ret <- .Call('dynIRT_estimate',
                 PACKAGE = 'emIRTEDM',
                 .starts$m,           # item motion 'm' start values
                 .starts$s,           # item status quo 's' start values
                 .starts$x,
                 .starts$p,           # (N x T) matrix of propensity starting values
                 .data$rc,
                 .data$startlegis,
                 .data$endlegis,
                 .data$prevlegis,     # N x 1 column matrix of previous contiguous legislators (== 0 when no prior contiguous legislators).
                 .data$bill.session,
                 .data$T,
                 .data$sponsor_index, # J-length vector of EDM sponsors
                 .data$anchor_group,  # J x 1 (0 = singleton; >0 = tied)
                 .priors$x.mu0,
                 .priors$x.sigma0,
                 .priors$x.sign,      # N x 1 legislator sign constraints (left-wing -> non-positive, right-wing -> non-negative, unconstrained)
                 .priors$item.sigma,  # Prior covariance matrix for s_{jt} and m_{jt} (centered on sponsor's x_{it}) 
                 .priors$omega2,
                 .priors$rho_p,       # AR(1) coefficient in (-1,1). Use 1.0 for random walk.
                 .priors$sig2_p,      # N x 1 innovation variance for propensity per legislator
                 .priors$pmu0,        # N x 1 prior mean for p at first served period (often 0 and large).
                 .priors$psigma0,     # N x 1 prior var for p at first served period (often 0 and large).
                 ifelse(!is.null(.control$threads), .control$threads, default$threads),
                 ifelse(!is.null(.control$verbose), .control$verbose, default$verbose),
                 ifelse(!is.null(.control$maxit), .control$maxit, default$maxit),
                 ifelse(!is.null(.control$thresh), .control$thresh, default$thresh),
                 ifelse(!is.null(.control$checkfreq), .control$checkfreq, default$checkfreq)
                 )

    cat(paste("\t",
              "Done in ",
              ret$runtime$iters,
              " iterations, using ",
              ret$runtime$threads,
              " threads.",
              "\n",
              sep = ""
              )
        )

	rownames(ret$means$x) <- rownames(.data$rc)
	rownames(ret$vars$x) <- rownames(.data$rc)

	rownames(ret$means$alpha) <- colnames(.data$rc)
	rownames(ret$means$beta) <- colnames(.data$rc)

	rownames(ret$vars$alpha) <- colnames(.data$rc)
	rownames(ret$vars$beta) <- colnames(.data$rc)

    cat(divider)
    ret$call <- cl
    class(ret) <- "dynIRT"
    return(ret)
}
