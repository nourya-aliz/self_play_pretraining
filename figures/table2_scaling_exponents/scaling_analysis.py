"""Shared compute-frontier construction and power-law fits.

The paper figure fits the asymptotic law ``loss = E + A * C**(-alpha)`` with
SciPy nonlinear least squares and plots the raw validation loss.
Point selection is shared by all fits; ties at exactly the same compute are
resolved by keeping the lowest loss before constructing each Pareto frontier.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, minimize


POOL = 1536
CTX = 4096


@dataclass(frozen=True)
class LogPowerFit:
    """Two-parameter finite-range power-law fit."""

    amplitude: float
    alpha: float
    log_r2: float
    log_rmse: float
    n_points: int

    def predict(self, compute):
        compute = np.asarray(compute, dtype=float)
        return self.amplitude * compute ** (-self.alpha)


@dataclass(frozen=True)
class FloorPowerFit:
    """Robust fit of ``loss = floor + amplitude * compute**(-alpha)``."""

    amplitude: float
    alpha: float
    floor: float
    rmse: float
    n_points: int

    def predict(self, compute):
        compute = np.asarray(compute, dtype=float)
        return self.floor + self.amplitude * compute ** (-self.alpha)


def pareto_front(points):
    """Return the strictly improving lower envelope of ``(compute, loss)``."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("points must be a non-empty array of (compute, loss)")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("compute and loss must be finite and positive")

    # np.argsort(compute) alone leaves equal-compute ordering unspecified.
    # Sorting loss second ensures that the best tied point is considered first.
    order = np.lexsort((values[:, 1], values[:, 0]))
    values = values[order]
    best = np.inf
    keep = []
    for compute, loss in values:
        if loss < best:
            best = loss
            keep.append((compute, loss))
    return np.asarray(keep, dtype=float)


def compute_frontier(
    data,
    metric,
    *,
    max_k=None,
    checkpoint_stride=1,
    checkpoint_keep=None,
    omit_rungs=(),
    pool=POOL,
    context=CTX,
):
    """Build the cross-rung envelope used by Figure 1.

    First a Pareto front is built within each model rung over checkpoint and
    ensemble size K.  A second Pareto pass produces the global compute
    envelope.  ``checkpoint_stride`` retains every nth eligible checkpoint
    within each rung.  ``checkpoint_keep`` can instead supply a mapping from
    rung name to the round indices retained in a resampling draw.
    """

    if checkpoint_stride < 1:
        raise ValueError("checkpoint_stride must be at least 1")
    omitted = set(omit_rungs)
    rung_fronts = []

    for rung, rung_data in sorted(data.items(), key=lambda item: item[1]["N"]):
        if rung.startswith("_") or rung in omitted:
            continue
        rounds = sorted(
            (int(round_string), row)
            for round_string, row in rung_data["rounds"].items()
            if int(round_string) > 0 and metric in row
        )
        if checkpoint_keep is not None:
            retained = set(checkpoint_keep[rung])
            rounds = [(round_index, row) for round_index, row in rounds if round_index in retained]
        rounds = rounds[::checkpoint_stride]
        points = []
        for round_index, row in rounds:
            for k_string, loss in row[metric].items():
                k = int(k_string)
                if max_k is not None and k > max_k:
                    continue
                compute = k * rung_data["N"] * pool * context * (round_index + 1.0)
                points.append((compute, loss))
        if points:
            rung_fronts.append(pareto_front(points))

    if not rung_fronts:
        raise ValueError(f"no points available for metric {metric!r}")
    return pareto_front(np.vstack(rung_fronts))


def fit_log_power_law(points):
    """Fit the finite-range law by OLS in log space (diagnostic)."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("at least three (compute, loss) points are required")
    log_compute = np.log(points[:, 0])
    log_loss = np.log(points[:, 1])
    slope, intercept = np.polyfit(log_compute, log_loss, 1)
    prediction = intercept + slope * log_compute
    residual = log_loss - prediction
    total = log_loss - log_loss.mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total)
    return LogPowerFit(
        amplitude=float(np.exp(intercept)),
        alpha=float(-slope),
        log_r2=float(r2),
        log_rmse=float(np.sqrt(np.mean(residual**2))),
        n_points=len(points),
    )


def _power_prediction_scaled(compute_scaled, amplitude_at_reference, alpha):
    """Pure power law parameterized at a well-conditioned reference compute."""

    return amplitude_at_reference * np.asarray(compute_scaled, dtype=float) ** (-alpha)


def fit_power_law(points):
    """Fit ``loss = A * compute**(-alpha)`` with SciPy nonlinear least squares.

    This follows the raw-loss ``scipy.optimize.curve_fit`` recipe used by the
    comparison papers, while retaining Figure 1's two-parameter law (no fitted
    floor).  Compute is divided by its geometric mean for numerical conditioning;
    this reparameterizes the amplitude but does not change the fitted curve or
    exponent.  ``curve_fit`` uses its default initialization and no bounds.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("at least three (compute, loss) points are required")
    if np.any(~np.isfinite(points)) or np.any(points <= 0):
        raise ValueError("compute and loss must be finite and positive")

    compute = points[:, 0]
    loss = points[:, 1]
    reference = float(np.exp(np.mean(np.log(compute))))
    compute_scaled = compute / reference
    parameters, _ = curve_fit(
        _power_prediction_scaled,
        compute_scaled,
        loss,
        maxfev=100_000,
    )
    amplitude_at_reference, alpha = map(float, parameters)
    amplitude = amplitude_at_reference * reference**alpha
    prediction = amplitude * compute ** (-alpha)
    log_loss = np.log(loss)
    log_prediction = np.log(prediction)
    residual = log_loss - log_prediction
    total = log_loss - log_loss.mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total)
    return LogPowerFit(
        amplitude=amplitude,
        alpha=alpha,
        log_r2=r2,
        log_rmse=float(np.sqrt(np.mean(residual**2))),
        n_points=len(points),
    )


def _floor_power_prediction_scaled(
    compute_scaled, amplitude_at_reference, alpha, floor
):
    """Asymptotic power law parameterized at a reference compute."""

    compute_scaled = np.asarray(compute_scaled, dtype=float)
    return floor + amplitude_at_reference * compute_scaled ** (-alpha)


def fit_curve_power_law_with_floor(points, *, compute_scale=1e15):
    """Fit ``loss = E + A * C**(-alpha)`` with SciPy ``curve_fit``.

    This follows the Appendix H.2 recipe in Kim, Kotha et al.: nonlinear
    least squares on raw loss, initial parameters ``[1.0, 0.5, 2.0]``, and
    nonnegative bounds.  Compute is expressed in units of ``compute_scale``
    solely to condition the optimization; this changes the reported amplitude
    but not the fitted exponent, floor, or curve.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("at least three (compute, loss) points are required")
    if np.any(~np.isfinite(points)) or np.any(points <= 0):
        raise ValueError("compute and loss must be finite and positive")
    if not np.isfinite(compute_scale) or compute_scale <= 0:
        raise ValueError("compute_scale must be finite and positive")

    compute = points[:, 0]
    loss = points[:, 1]
    compute_scaled = compute / compute_scale
    parameters, _ = curve_fit(
        _floor_power_prediction_scaled,
        compute_scaled,
        loss,
        p0=[1.0, 0.5, 2.0],
        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        maxfev=100_000,
    )
    amplitude_at_reference, alpha, floor = map(float, parameters)
    amplitude = amplitude_at_reference * compute_scale**alpha
    prediction = floor + amplitude * compute ** (-alpha)
    return FloorPowerFit(
        amplitude=amplitude,
        alpha=alpha,
        floor=floor,
        rmse=float(np.sqrt(np.mean((prediction - loss) ** 2))),
        n_points=len(points),
    )


def _floor_power_prediction(log_compute, log_amplitude, exponent, log_floor_ratio):
    """Numerically stable base-10 form of ``A * C**exponent + E``."""

    power_term = exponent * log_compute
    maximum = np.maximum(power_term, log_floor_ratio)
    log_sum = maximum + np.log10(
        10.0 ** (power_term - maximum) + 10.0 ** (log_floor_ratio - maximum)
    )
    return 10.0 ** (log_amplitude + log_sum)


def _huber(residual, delta=1e-3):
    magnitude = np.abs(residual)
    return np.where(
        magnitude <= delta,
        0.5 * residual**2,
        delta * (magnitude - 0.5 * delta),
    )


def fit_power_law_with_floor(points):
    """Fit ``loss = E + A * C**(-alpha)`` with a robust (Huber) objective.

    The optimization matches ``plot_modality_frontier.py``: symmetric Huber
    loss in bits/byte and a nine-start Nelder-Mead search.  Fitting in this
    parameterization keeps the positive amplitude and floor numerically stable
    across the very large compute values on the x-axis.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 6:
        raise ValueError("at least six (compute, loss) points are required")
    if np.any(~np.isfinite(points)) or np.any(points <= 0):
        raise ValueError("compute and loss must be finite and positive")

    log_compute = np.log10(points[:, 0])
    loss = points[:, 1]
    if np.ptp(log_compute) < 1.5:
        raise ValueError("compute points must span at least 1.5 decades")

    def objective(parameters):
        prediction = _floor_power_prediction(log_compute, *parameters)
        return float(np.sum(_huber(prediction - loss)))

    best = None
    for amplitude_start in (math.log10(loss.max() * 3.0), 1.0, 2.0):
        for exponent_start in (-0.03, -0.1, -0.3):
            floor_ratio_start = (
                math.log10(max(loss.min() * 0.5, 1e-3)) - amplitude_start
            )
            result = minimize(
                objective,
                [amplitude_start, exponent_start, floor_ratio_start],
                method="Nelder-Mead",
                options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9},
            )
            if best is None or result.fun < best.fun:
                best = result

    log_amplitude, exponent, log_floor_ratio = best.x
    amplitude = 10.0**log_amplitude
    floor = 10.0 ** (log_amplitude + log_floor_ratio)
    prediction = _floor_power_prediction(
        log_compute, log_amplitude, exponent, log_floor_ratio
    )
    return FloorPowerFit(
        amplitude=float(amplitude),
        alpha=float(-exponent),
        floor=float(floor),
        rmse=float(np.sqrt(np.mean((prediction - loss) ** 2))),
        n_points=len(points),
    )
