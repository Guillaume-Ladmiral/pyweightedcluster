"""Do the two plot-payload reconstructions conceal a defect? (audit §9 item 2)

`RParitySession._clustrange_plot_payload` and `._clustrangefamily_plot_payload`
re-implement `plot.clustrange` and `plot.clustrangefamily`'s statistic selection
and normalisation **in R** rather than calling the installed S3 methods. The
2026-08-24 audit adjudicated both as "material, unquantified": a Python defect
that copied the same misreading would pass unnoticed. That is the same class of
risk as finding F-03, which turned out to hide an S1.

This module settles it. The installed S3 methods are captured with their drawing
primitives replaced by recorders — the idiom
`RParitySession.seqclararange_ari_boxplot_data` already uses — so what R
*actually plots* is compared against what the reconstruction produces, on
tie-rich integer dissimilarities and across the normalisations, including
`norm="range"` and the `stat="all"` promotion.
"""

from __future__ import annotations

import numpy as np
import pytest
import rpy2.robjects as ro

from weightedcluster import as_clustrange, wcCmpCluster

pytest_plugins = ["tests.r_parity_utils"]


def _tie_rich_dissimilarity(seed: int = 5, n_seq: int = 10, n_time: int = 6, n_states: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, n_states, size=(n_seq, n_time))
    D = (X[:, None, :] != X[None, :, :]).sum(axis=2).astype(float)
    upper = D[np.triu_indices(n_seq, k=1)]
    assert np.unique(upper).size < upper.size, "fixture must contain ties"
    return D


_RECORDERS = """
  fn <- getS3method('plot', '{cls}')
  env <- new.env(parent = environment(fn))
  env$rec_y <- list()
  env$rec_labels <- NULL
  evalq(plot <- function(...) invisible(NULL), envir = env)
  evalq(lines <- function(x, y, ...) {{
      rec_y[[length(rec_y) + 1L]] <<- y
      invisible(NULL)
  }}, envir = env)
  evalq(polygon <- function(...) invisible(NULL), envir = env)
  evalq(legend <- function(...) {{ invisible(NULL) }}, envir = env)
  evalq(layout <- function(...) invisible(NULL), envir = env)
  evalq(par <- function(...) invisible(list()), envir = env)
  evalq(crflegend <- function(...) invisible(NULL), envir = env)
  environment(fn) <- env
"""


def _record_real_plot(session, result_name: str, cls: str, call_args: str):
    """Run the installed S3 plot method with recorders and return its curves."""
    session.r(f"""
        wc_plot_capture <- local({{
          {_RECORDERS.format(cls=cls)}
          grDevices::pdf(NULL)
          on.exit(grDevices::dev.off())
          error <- tryCatch({{
            fn(get('{result_name}', envir = .GlobalEnv){call_args})
            NULL
          }}, error = function(e) conditionMessage(e))
          list(y = env$rec_y, error = error)
        }})
        """)
    error = session.r("wc_plot_capture$error")
    if error is not ro.NULL:
        return None, str(error[0])
    count = int(session.r("length(wc_plot_capture$y)")[0])
    curves = [
        np.asarray(session.r(f"as.numeric(wc_plot_capture$y[[{index + 1}]])"), dtype=float)
        for index in range(count)
    ]
    return curves, None


@pytest.mark.parametrize(
    ("stat", "norm"),
    [
        ("noCH", "none"),
        ("noCH", "range"),
        ("noCH", "zscore"),
        ("all", "none"),
        ("all", "range"),
    ],
)
def test_clustrange_plot_reconstruction_matches_the_installed_s3_method(
    r_parity_session, stat: str, norm: str
):
    D = _tie_rich_dissimilarity()
    family = wcCmpCluster(D, maxcluster=5, method="complete", pam_combine=False)
    columns = {
        name: np.asarray(values) for name, values in family["complete"]["clustering"].items()
    }
    as_clustrange(columns, D)  # keep the Python side exercised on the same input

    r_parity_session._set_distance_matrix(D)
    r_parity_session.r("wc_oq2_obj <- NULL")
    for key, values in columns.items():
        r_parity_session.r.assign("tmpcol", ro.IntVector(np.asarray(values, dtype=int)))
        r_parity_session.r.assign("tmpname", ro.StrVector([str(key)]))
        r_parity_session.r(
            "if (is.null(wc_oq2_obj)) { wc_oq2_obj <- data.frame(tmp = as.integer(tmpcol)); "
            "names(wc_oq2_obj) <- as.character(tmpname) } else { "
            "wc_oq2_obj[[as.character(tmpname)]] <- as.integer(tmpcol) }"
        )
    r_parity_session.r(
        "wc_oq2_range <- WeightedCluster::as.clustrange(wc_oq2_obj, diss = as.dist(D), R = 1)"
    )

    reconstructed = r_parity_session._clustrange_plot_payload("wc_oq2_range", stat=stat, norm=norm)
    curves, error = _record_real_plot(
        r_parity_session,
        "wc_oq2_range",
        "clustrange",
        f", stat='{stat}', norm='{norm}'",
    )

    assert error is None, error
    values = np.asarray(reconstructed["values"], dtype=float)
    assert len(curves) == values.shape[1], (len(curves), values.shape)
    for index, curve in enumerate(curves):
        np.testing.assert_allclose(
            curve,
            values[:, index],
            rtol=0.0,
            atol=1e-12,
            err_msg=f"{stat}/{norm} column {index} " f"({reconstructed['stat_names'][index]})",
        )


@pytest.mark.parametrize("norm", ["none", "range", "zscore"])
def test_clustrangefamily_plot_reconstruction_matches_the_installed_s3_method(
    r_parity_session, norm: str
):
    D = _tie_rich_dissimilarity(seed=9)
    r_parity_session._set_distance_matrix(D)
    r_parity_session.r(
        "wc_oq2_family <- WeightedCluster::wcCmpCluster(D, maxcluster = 5, "
        "method = c('complete', 'average'), pam.combine = FALSE)"
    )

    reconstructed = r_parity_session._clustrangefamily_plot_payload(
        "wc_oq2_family", stat="ASW", norm=norm
    )
    curves, error = _record_real_plot(
        r_parity_session,
        "wc_oq2_family",
        "clustrangefamily",
        f", stat='ASW', norm='{norm}'",
    )

    assert error is None, error
    assert reconstructed["data_error"] is None, reconstructed["data_error"]
    # group="stat" gives a (kvals, method, stat) array: one panel per statistic,
    # one plotted curve per method inside it.
    values = np.asarray(reconstructed["values"], dtype=float)
    assert values.ndim == 3, values.shape
    assert values.shape[1:] == (len(reconstructed["methods"]), len(reconstructed["stats"]))
    assert len(curves) == values.shape[1], (len(curves), values.shape)
    for index, curve in enumerate(curves):
        np.testing.assert_allclose(
            curve,
            values[:, index, 0],
            rtol=0.0,
            atol=1e-12,
            err_msg=f"{norm} method {reconstructed['methods'][index]}",
        )
