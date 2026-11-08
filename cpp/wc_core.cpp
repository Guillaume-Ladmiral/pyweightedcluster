#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <map>
#include <string>

namespace py = pybind11;

// Helper: check square matrix
static inline void check_square(const py::array_t<double>& D) {
    if (D.ndim() != 2 || D.shape(0) != D.shape(1)) {
        throw std::invalid_argument("diss must be a square (N,N) float array");
    }
    if (D.shape(0) == 0) {
        throw std::invalid_argument("diss must be non-empty");
    }
    auto values = D.unchecked<2>();
    for (py::ssize_t i = 0; i < D.shape(0); ++i) {
        for (py::ssize_t j = 0; j < D.shape(1); ++j) {
            const double value = values(i, j);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("diss must contain only finite distances");
            }
            if (value < 0.0) {
                throw std::invalid_argument("diss distances must be non-negative");
            }
        }
    }
}

static std::string array_kind(const py::array& values) {
    return py::str(values.dtype().attr("kind")).cast<std::string>();
}

static std::vector<double> checked_weights(py::object weights_obj, py::ssize_t N) {
    std::vector<double> weights((size_t)N, 1.0);
    if (weights_obj.is_none()) return weights;

    py::array raw = py::array::ensure(weights_obj);
    if (!raw || raw.ndim() != 1 || raw.shape(0) != N) {
        throw std::invalid_argument("weights must be length N");
    }
    const std::string kind = array_kind(raw);
    if (kind != "b" && kind != "i" && kind != "u" && kind != "f") {
        throw std::invalid_argument("weights must contain numeric values");
    }
    auto converted = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(raw);
    if (!converted) throw std::invalid_argument("weights must contain numeric values");
    auto values = converted.unchecked<1>();
    for (py::ssize_t i = 0; i < N; ++i) {
        const double value = values(i);
        if (!std::isfinite(value)) {
            throw std::invalid_argument("weights must contain only finite values");
        }
        weights[(size_t)i] = std::max(0.0, value);
    }
    return weights;
}

struct CheckedLabels {
    std::vector<int> values;
    int count;
};

static CheckedLabels checked_labels(py::object labels_obj, py::ssize_t N) {
    py::array raw = py::array::ensure(labels_obj);
    if (!raw || raw.ndim() != 1 || raw.shape(0) != N) {
        throw std::invalid_argument("labels must be length N");
    }
    const std::string kind = array_kind(raw);
    if (kind != "i" && kind != "u") {
        throw std::invalid_argument("labels must contain integers");
    }
    auto converted =
        py::array_t<long long, py::array::c_style | py::array::forcecast>::ensure(raw);
    if (!converted) throw std::invalid_argument("labels must contain integers");
    auto input = converted.unchecked<1>();

    std::vector<int> labels((size_t)N);
    std::vector<char> seen((size_t)N, 0);
    int max_label = -1;
    for (py::ssize_t i = 0; i < N; ++i) {
        const long long value = input(i);
        if (value < 0 || value >= N) {
            throw std::invalid_argument("labels must be contiguous zero-based integers");
        }
        labels[(size_t)i] = static_cast<int>(value);
        seen[(size_t)value] = 1;
        max_label = std::max(max_label, static_cast<int>(value));
    }
    for (int label = 0; label <= max_label; ++label) {
        if (!seen[(size_t)label]) {
            throw std::invalid_argument("labels must be contiguous zero-based integers");
        }
    }
    return {std::move(labels), max_label + 1};
}

// Assign each point to nearest medoid; return labels (0..k-1) and objective value (weighted sum)
static std::pair<std::vector<int>, double> assign_and_objective(
    const py::array_t<double>& D, const std::vector<int>& medoids, const std::vector<double>& w
) {
    const py::ssize_t N = D.shape(0);
    auto Dbuf = D.unchecked<2>();
    std::vector<int> labels(N, 0);
    double obj = 0.0;
    for (py::ssize_t i = 0; i < N; ++i) {
        double best = std::numeric_limits<double>::infinity();
        int best_j = 0;
        for (size_t m = 0; m < medoids.size(); ++m) {
            const double d = Dbuf(i, medoids[m]);
            if (d < best) { best = d; best_j = static_cast<int>(m); }
        }
        labels[(size_t)i] = best_j;
        obj += w[(size_t)i] * best;
    }
    return {labels, obj};
}

// PAM-like weighted k-medoids with SWAP improvements
static py::tuple weighted_medoids(
    const py::array_t<double>& D,
    int k,
    py::object weights_obj,
    int max_iter,
    int n_init,
    int random_state,
    py::object init_medoids_obj = py::none()
) {
    check_square(D);
    if (k < 2) throw std::invalid_argument("k must be >= 2");
    const py::ssize_t N = D.shape(0);
    if (k > N) throw std::invalid_argument("k cannot exceed N");

    std::vector<double> w = checked_weights(weights_obj, N);

    // RNG
    std::mt19937_64 rng(static_cast<uint64_t>(random_state < 0 ? 0 : random_state));

    std::vector<int> best_medoids;
    std::vector<int> best_labels;
    double best_obj = std::numeric_limits<double>::infinity();

    const int runs = std::max(1, n_init);
    // Prepare optional user-provided initialization
    std::vector<int> user_init;
    if (!init_medoids_obj.is_none()) {
        py::array raw = py::array::ensure(init_medoids_obj);
        if (!raw || raw.ndim() != 1 || raw.shape(0) != k) {
            throw std::invalid_argument("init_medoids must be shape (k,)");
        }
        const std::string kind = array_kind(raw);
        if (kind != "i" && kind != "u") {
            throw std::invalid_argument("init_medoids must contain integers");
        }
        auto init_in =
            py::array_t<long long, py::array::c_style | py::array::forcecast>::ensure(raw);
        if (!init_in) throw std::invalid_argument("init_medoids must contain integers");
        auto ib = init_in.unchecked<1>();
        std::vector<char> seen((size_t)N, 0);
        for (int i = 0; i < k; ++i) {
            const long long value = ib((size_t)i);
            if (value < 0 || value >= N) {
                throw std::invalid_argument("init_medoids out of range");
            }
            const int v = static_cast<int>(value);
            if (seen[(size_t)v]) throw std::invalid_argument("init_medoids must be unique");
            seen[(size_t)v] = 1;
            user_init.push_back(v);
        }
    }

    // The ordinary C++ accumulator cannot preserve ordering once a finite
    // weight-distance product enters the subnormal range (nor when a product
    // or its reduction leaves binary64 range). Delegate those exceptional
    // inputs to the exact dyadic implementation used by the public facade.
    auto Dbuf = D.unchecked<2>();
    bool exact_objectives = false;
    int minimum_product_exponent = std::numeric_limits<int>::max();
    int maximum_product_exponent = std::numeric_limits<int>::min();
    const int reduction_count = std::max<int>(1, static_cast<int>(N));
    const int reduction_rounding_window =
        52 - static_cast<int>(std::ceil(std::log2(static_cast<double>(reduction_count))));
    const int overflow_product_exponent =
        std::ilogb(std::numeric_limits<double>::max() / reduction_count) + 1;
    for (py::ssize_t i = 0; i < N && !exact_objectives; ++i) {
        if (w[(size_t)i] <= 0.0) continue;
        if (w[(size_t)i] > std::numeric_limits<double>::max() / reduction_count ||
            w[(size_t)i] < std::numeric_limits<double>::min()) {
            exact_objectives = true;
            break;
        }
        for (py::ssize_t j = 0; j < N; ++j) {
            const double distance = Dbuf(i, j);
            if (distance <= 0.0) continue;
            const double product = w[(size_t)i] * distance;
            if (product == 0.0 || !std::isfinite(product) ||
                product < std::numeric_limits<double>::min()) {
                exact_objectives = true;
                break;
            }
            const int product_exponent =
                std::ilogb(w[(size_t)i]) + std::ilogb(distance) + 2;
            minimum_product_exponent = std::min(minimum_product_exponent, product_exponent);
            maximum_product_exponent = std::max(maximum_product_exponent, product_exponent);
        }
    }
    if (!exact_objectives && minimum_product_exponent != std::numeric_limits<int>::max()) {
        exact_objectives =
            maximum_product_exponent >= overflow_product_exponent ||
            maximum_product_exponent - minimum_product_exponent >= reduction_rounding_window;
    }
    if (exact_objectives) {
        py::array_t<double> weight_array({N});
        auto weight_buffer = weight_array.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < N; ++i) weight_buffer(i) = w[(size_t)i];
        return py::module_::import("weightedcluster._fallback")
            .attr("weighted_medoids")(
                D, k, weight_array, max_iter, n_init, random_state, init_medoids_obj
            )
            .cast<py::tuple>();
    }

    for (int run = 0; run < runs; ++run) {
        std::vector<int> medoids;
        if (!user_init.empty() && run == 0) {
            // Use user-provided initialization for first run
            medoids = user_init;
        } else {
            // Initialize medoids by random unique indices
            std::vector<int> idx((size_t)N);
            std::iota(idx.begin(), idx.end(), 0);
            if (random_state >= 0) {
                std::shuffle(idx.begin(), idx.end(), rng);
            } else {
                // deterministic but still diverse with simple stride
                std::rotate(idx.begin(), idx.begin() + (run % std::max(1, (int)N)), idx.end());
            }
            medoids.assign(idx.begin(), idx.begin() + k);
        }

        // SWAP phase
        const int iters = std::max(1, max_iter);
        bool improved = true;
        int it = 0;
        // WeightedCluster 2.0 PAMonce best-improvement swap. The candidate
        // guard and strict loop-order tie rule are part of the scientific
        // result, not optimization details.
        auto Db = D.unchecked<2>();
        double max_distance = 0.0;
        for (py::ssize_t i = 0; i < N; ++i) {
            for (py::ssize_t j = i + 1; j < N; ++j) {
                max_distance = std::max(max_distance, Db(i, j));
            }
        }
        max_distance = 1.1 * max_distance + 1.0;
        std::vector<int> labels((size_t)N, 0);
        std::vector<double> nearest((size_t)N, max_distance);
        std::vector<double> second((size_t)N, max_distance);
        auto compute_nearest_two = [&]() {
            std::fill(nearest.begin(), nearest.end(), max_distance);
            std::fill(second.begin(), second.end(), max_distance);
            for (py::ssize_t observation = 0; observation < N; ++observation) {
                for (int slot = 0; slot < k; ++slot) {
                    const double distance = Db(observation, medoids[(size_t)slot]);
                    if (nearest[(size_t)observation] > distance) {
                        second[(size_t)observation] = nearest[(size_t)observation];
                        nearest[(size_t)observation] = distance;
                        labels[(size_t)observation] = slot;
                    } else if (second[(size_t)observation] > distance) {
                        second[(size_t)observation] = distance;
                    }
                }
            }
        };
        double obj = -1.0;
        while (improved && it < iters) {
            improved = false;
            ++it;
            compute_nearest_two();
            if (obj < 0.0) {
                obj = 0.0;
                for (py::ssize_t observation = 0; observation < N; ++observation) {
                    obj += w[(size_t)observation] * nearest[(size_t)observation];
                }
            }
            double best_delta = 1.0;
            int best_mi = -1;
            py::ssize_t best_h = -1;
            for (int mi = 0; mi < k; ++mi) {
                const int current = medoids[(size_t)mi];
                double removal_cost = 0.0;
                std::vector<double> fallback((size_t)N, 0.0);
                for (py::ssize_t observation = 0; observation < N; ++observation) {
                    if (labels[(size_t)observation] == mi) {
                        removal_cost += w[(size_t)observation] *
                            (second[(size_t)observation] - nearest[(size_t)observation]);
                        fallback[(size_t)observation] = second[(size_t)observation];
                    } else {
                        fallback[(size_t)observation] = nearest[(size_t)observation];
                    }
                }
                for (py::ssize_t h = 0; h < N; ++h) {
                    if (Db(current, h) <= 0.0) continue;
                    double addition_gain = removal_cost;
                    for (py::ssize_t observation = 0; observation < N; ++observation) {
                        const double candidate_distance = Db(h, observation);
                        if (candidate_distance < fallback[(size_t)observation]) {
                            addition_gain += w[(size_t)observation] *
                                (candidate_distance - fallback[(size_t)observation]);
                        }
                    }
                    if (best_delta > addition_gain) {
                        best_delta = addition_gain;
                        best_mi = mi;
                        best_h = h;
                    }
                }
            }
            // -1e-10 is R's WEIGHTED_CLUST_TOL (WeightedCluster src/PAMonce.h:5).
            // The Python sites share it as PAMONCE_SWAP_TOLERANCE in
            // src/weightedcluster/_constants.py; a C++ translation unit cannot
            // import that, so this literal must be kept in step with it by hand.
            if (best_mi >= 0 && best_delta < -1e-10) {
                medoids[(size_t)best_mi] = static_cast<int>(best_h);
                obj += best_delta;
                improved = true;
            }
        }
        compute_nearest_two();
        if (best_medoids.empty() || obj < best_obj) {
            best_obj = obj;
            best_medoids = medoids;
            best_labels = labels;
        }
    }

    // Return labels (0-based), medoid indices (0-based), objective
    py::array_t<int> labels_arr({(py::ssize_t)N});
    auto lb = labels_arr.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < N; ++i) lb(i) = best_labels[(size_t)i];

    py::array_t<int> med_arr({(py::ssize_t)k});
    auto mb = med_arr.mutable_unchecked<1>();
    for (int i = 0; i < k; ++i) mb(i) = best_medoids[i];

    return py::make_tuple(labels_arr, med_arr, best_obj);
}

// Fast cluster quality stats (pairwise-based). Returns (stats dict, asw dict, asw_w dict)
static py::tuple cluster_quality(const py::array_t<double>& D, py::object labels_obj, py::object weights_obj) {
    check_square(D);
    const py::ssize_t N = D.shape(0);
    CheckedLabels checked = checked_labels(labels_obj, N);
    const std::vector<int>& labels = checked.values;
    const int k = checked.count;
    std::vector<double> w = checked_weights(weights_obj, N);
    auto Db = D.unchecked<2>();
    double distance_scale = 0.0;
    double minimum_positive_distance = std::numeric_limits<double>::infinity();
    std::size_t positive_distance_count = 0;
    for (py::ssize_t i = 0; i < N; ++i) {
        for (py::ssize_t j = 0; j < N; ++j) {
            if (w[(size_t)i] <= 0.0 || w[(size_t)j] <= 0.0) continue;
            const double distance = Db(i, j);
            distance_scale = std::max(distance_scale, distance);
            if (distance > 0.0) {
                minimum_positive_distance = std::min(minimum_positive_distance, distance);
                ++positive_distance_count;
            }
        }
    }
    const double square_min = std::sqrt(std::numeric_limits<double>::min());
    const double square_max = std::sqrt(
        std::numeric_limits<double>::max() /
        std::max<std::size_t>(1, positive_distance_count));
    const int distance_rounding_window =
        52 - static_cast<int>(std::ceil(std::log2(
                 std::max(1.0, static_cast<double>(positive_distance_count)))));
    const bool distance_exponent_spread_loss =
        positive_distance_count > 0 &&
        2 * (std::ilogb(distance_scale) -
             std::ilogb(minimum_positive_distance)) >= distance_rounding_window;
    const bool square_outside_range =
        std::isfinite(distance_scale) && distance_scale > 0.0 &&
        (minimum_positive_distance < square_min || distance_scale > square_max ||
         distance_exponent_spread_loss);
    if (!square_outside_range) distance_scale = 1.0;

    std::size_t positive_weight_count = 0;
    double minimum_positive_weight = std::numeric_limits<double>::infinity();
    double maximum_positive_weight = 0.0;
    for (double weight : w) {
        if (weight <= 0.0) continue;
        ++positive_weight_count;
        minimum_positive_weight = std::min(minimum_positive_weight, weight);
        maximum_positive_weight = std::max(maximum_positive_weight, weight);
    }
    const double pair_square_min = std::sqrt(std::numeric_limits<double>::min());
    const double positive_weight_count_d =
        std::max(1.0, static_cast<double>(positive_weight_count));
    const double pair_count_bound =
        positive_weight_count_d * positive_weight_count_d;
    const double pair_aggregate_max = std::sqrt(
        std::numeric_limits<double>::max() / (2.0 * pair_count_bound));
    const double pair_second_order_min =
        std::sqrt(std::sqrt(std::numeric_limits<double>::min()));
    const double pair_second_order_max = std::sqrt(std::sqrt(
        std::numeric_limits<double>::max() /
        (4.0 * pair_count_bound * pair_count_bound)));
    const int pair_rounding_window =
        52 - static_cast<int>(std::ceil(std::log2(pair_count_bound)));
    const bool pair_exponent_spread_loss =
        positive_weight_count > 0 &&
        2 * (std::ilogb(maximum_positive_weight) -
             std::ilogb(minimum_positive_weight)) >= pair_rounding_window;
    const bool pair_mass_outside_range =
        positive_weight_count > 0 &&
        (minimum_positive_weight < pair_square_min ||
         maximum_positive_weight > pair_aggregate_max ||
         minimum_positive_weight < pair_second_order_min ||
         maximum_positive_weight > pair_second_order_max ||
         pair_exponent_spread_loss);
    bool exact_positive_fallback = square_outside_range || pair_mass_outside_range;
    for (double weight : w) {
        if (weight <= 0.0) continue;
        const double square = weight * weight;
        if (square == 0.0 || !std::isfinite(square)) {
            exact_positive_fallback = true;
            break;
        }
    }
    if (!exact_positive_fallback && distance_scale != 1.0) {
        for (py::ssize_t i = 0; i < N && !exact_positive_fallback; ++i) {
            for (py::ssize_t j = 0; j < N; ++j) {
                const double distance = Db(i, j);
                if (distance > 0.0 && distance / distance_scale == 0.0) {
                    exact_positive_fallback = true;
                    break;
                }
            }
        }
    }
    if (exact_positive_fallback) {
        return py::module_::import("weightedcluster._fallback")
            .attr("cluster_quality")(D, labels_obj, weights_obj)
            .cast<py::tuple>();
    }

    struct KendallBucket {
        double clustDist0{0.0}; // within-cluster pairs
        double clustDist1{0.0}; // between-cluster pairs
    };

    std::vector<double> sizes((size_t)k, 0.0);
    std::vector<double> errors((size_t)k, 0.0);
    std::vector<double> errors2((size_t)k, 0.0);
    for (py::ssize_t i = 0; i < N; ++i) sizes[(size_t)labels[(size_t)i]] += w[(size_t)i];

    std::map<double, KendallBucket> kendall;
    kendall[0.0] = KendallBucket{};

    double totweights = 0.0;
    double wxy = 0.0;
    double wx = 0.0;
    double wy = 0.0;
    double wx2 = 0.0;

    for (py::ssize_t i = 0; i < N; ++i) {
        if (w[(size_t)i] <= 0.0) continue;
        int ci = labels[(size_t)i];
        double ww = w[(size_t)i] * w[(size_t)i];
        wy += ww;
        kendall[0.0].clustDist0 += ww;
        totweights += ww;

        for (py::ssize_t j = i + 1; j < N; ++j) {
            if (w[(size_t)j] <= 0.0) continue;
            ww = 2.0 * w[(size_t)i] * w[(size_t)j];
            double xx = Db(i, j) / distance_scale;
            KendallBucket& bucket = kendall[xx];
            double xw = ww * xx;
            double xxw = xw * xx;
            wx += xw;
            wx2 += xxw;
            if (labels[(size_t)i] == labels[(size_t)j]) {
                errors[(size_t)ci] += xw;
                errors2[(size_t)ci] += xxw;
                wxy += xw;
                wy += ww;
                bucket.clustDist0 += ww;
            } else {
                bucket.clustDist1 += ww;
            }
            totweights += ww;
        }
    }

    double xb = wx / totweights;
    double yb = wy / totweights;
    double covx = wx2 / totweights - xb * xb;
    double covy = wy / totweights - yb * yb;
    double covxy = wxy / totweights - yb * xb;
    double pearson =
        (covx > 0.0 && covy > 0.0) ? (covxy / std::sqrt(covx * covy)) : std::numeric_limits<double>::quiet_NaN();
    double PBC = -pearson;

    double nc = 0.0;
    double nd = 0.0;
    double currentclustdist0 = 0.0;
    double currentclustdist1 = 0.0;
    double totdist0 = wy;
    double totdist1 = std::max(0.0, totweights - wy);
    double ntiesdist = 0.0;
    double Smin = 0.0;
    double wSmin = wy;
    double Smax = 0.0;
    double wSmax = totdist1;
    double currentww = 0.0;
    for (auto it = kendall.begin(); it != kendall.end(); ++it) {
        const KendallBucket& b = it->second;
        double ww = b.clustDist0 + b.clustDist1;
        if (ww <= 0.0) continue;

        if (currentww <= wSmin) {
            if (currentww + ww > wSmin) {
                Smin += (wSmin - currentww) * it->first;
            } else {
                Smin += ww * it->first;
            }
        }
        currentww += ww;
        if (currentww > wSmax) {
            if (currentww - ww < wSmax) {
                Smax += (currentww - wSmax) * it->first;
            } else {
                Smax += ww * it->first;
            }
        }

        ntiesdist += b.clustDist1 * b.clustDist0;
        nc += b.clustDist1 * currentclustdist0;
        nd += b.clustDist0 * currentclustdist1;
        currentclustdist0 += b.clustDist0;
        currentclustdist1 += b.clustDist1;
        nc += b.clustDist0 * std::max(0.0, totdist1 - currentclustdist1);
        nd += b.clustDist1 * std::max(0.0, totdist0 - currentclustdist0);
    }
    double HG = ((nc + nd) > 0.0) ? ((nc - nd) / (nc + nd)) : std::numeric_limits<double>::quiet_NaN();
    double HGSD =
        ((nc + nd + ntiesdist) > 0.0) ? ((nc - nd) / (nc + nd + ntiesdist)) : std::numeric_limits<double>::quiet_NaN();
    double HC = (Smax > Smin) ? ((wxy - Smin) / (Smax - Smin)) : std::numeric_limits<double>::quiet_NaN();

    double SSres = 0.0;
    double SS2res = 0.0;
    totweights = 0.0;
    for (int c = 0; c < k; ++c) {
        double sz = sizes[(size_t)c];
        SSres += errors[(size_t)c] / sz;
        SS2res += errors2[(size_t)c] / sz;
        totweights += sz;
    }

    double SSexpl = wx / totweights - SSres;
    double SS2expl = wx2 / totweights - SS2res;
    double dncluster = (double)k;
    double CH = (SSexpl / (dncluster - 1.0)) / (SSres / (totweights - dncluster));
    double R2 = SSexpl / (SSres + SSexpl);
    double CHsq = (SS2expl / (dncluster - 1.0)) / (SS2res / (totweights - dncluster));
    double R2sq = SS2expl / (SS2res + SS2expl);

    std::vector<double> tmp((size_t)k, 0.0);
    std::vector<double> asw_cluster_num((size_t)k, 0.0);
    std::vector<double> asww_cluster_num((size_t)k, 0.0);
    double asw_i = 0.0;
    double asw_w = 0.0;
    for (py::ssize_t i = 0; i < N; ++i) {
        if (w[(size_t)i] <= 0.0) continue;
        int ci = labels[(size_t)i];
        double aik = 0.0;
        std::fill(tmp.begin(), tmp.end(), 0.0);
        for (py::ssize_t j = 0; j < N; ++j) {
            if (i == j) continue;
            double contrib = w[(size_t)j] * (Db(i, j) / distance_scale);
            if (labels[(size_t)j] == ci) {
                aik += contrib;
            } else {
                tmp[(size_t)labels[(size_t)j]] += contrib;
            }
        }

        double bik = std::numeric_limits<double>::max();
        for (int c = 0; c < k; ++c) {
            if (c == ci) continue;
            double candidate = tmp[(size_t)c] / sizes[(size_t)c];
            if (bik >= candidate) bik = candidate;
        }

        double aik_w =
            (sizes[(size_t)ci] > 0.0) ? (aik / sizes[(size_t)ci]) : std::numeric_limits<double>::quiet_NaN();
        if (sizes[(size_t)ci] <= 1.0) {
            aik = 0.0;
        } else {
            aik /= (sizes[(size_t)ci] - 1.0);
        }

        double denom_i = std::max(aik, bik);
        double denom_w = std::max(aik_w, bik);
        double sik_i = w[(size_t)i] * ((bik - aik) / denom_i);
        double sik_w = w[(size_t)i] * ((bik - aik_w) / denom_w);
        asw_cluster_num[(size_t)ci] += sik_i;
        asww_cluster_num[(size_t)ci] += sik_w;
        asw_i += sik_i;
        asw_w += sik_w;
    }

    py::dict asw;
    py::dict asw_w_dict;
    const bool asw_undefined = std::any_of(
        sizes.begin(), sizes.end(), [](double size) { return size < 1.0; }
    );
    for (int c = 0; c < k; ++c) {
        double sz = sizes[(size_t)c];
        double c_asw =
            asw_undefined
                ? std::numeric_limits<double>::quiet_NaN()
                : asw_cluster_num[(size_t)c] / sz;
        double c_asww = (sz > 0.0) ? (asww_cluster_num[(size_t)c] / sz) : std::numeric_limits<double>::quiet_NaN();
        asw[py::int_(c + 1)] = py::float_(c_asw);
        asw_w_dict[py::int_(c + 1)] = py::float_(c_asww);
    }

    py::dict stats;
    stats["PBC"] = py::float_(PBC);
    stats["HG"] = py::float_(HG);
    stats["HGSD"] = py::float_(HGSD);
    stats["ASW"] = py::float_(
        asw_undefined ? std::numeric_limits<double>::quiet_NaN() : asw_i / totweights
    );
    stats["ASWw"] = py::float_(asw_w / totweights);
    stats["CH"] = py::float_(CH);
    stats["R2"] = py::float_(R2);
    stats["CHsq"] = py::float_(CHsq);
    stats["R2sq"] = py::float_(R2sq);
    stats["HC"] = py::float_(HC);

    return py::make_tuple(stats, asw, asw_w_dict);
}

PYBIND11_MODULE(_wc_core, m) {
    m.doc() = "WeightedCluster core algorithms (pybind11)";

    m.def("weighted_medoids", &weighted_medoids,
          py::arg("diss"), py::arg("k"), py::arg("weights") = py::none(),
          py::arg("max_iter") = 100, py::arg("n_init") = 1, py::arg("random_state") = 0,
          py::arg("init_medoids") = py::none(),
          R"pbdoc(Weighted k-medoids (PAM-like) with SWAP improvements.
Optionally start from user-provided initial medoids (indices).
Returns (labels_0based, medoids_idx_0based, objective).)pbdoc");

    m.def("cluster_quality", &cluster_quality, py::arg("diss"), py::arg("labels"), py::arg("weights") = py::none(),
          R"pbdoc(Fast pairwise-based cluster quality metrics and per-cluster ASW/ASWw.
Returns (stats: dict, ASW: dict, ASWw: dict).)pbdoc");
}
