import numpy as np


def inject_level_shift(data: np.ndarray, t_shift: int, magnitude: float) -> np.ndarray:
    """
    data: shape (T, n_vars) or (T,)
    t_shift: index where shift starts
    magnitude: how many train-std units to shift by
    Returns perturbed copy of data (original not modified).
    """
    out = data.copy()
    out[t_shift:] += magnitude
    return out


def inject_variance_shift(data: np.ndarray, t_shift: int, alpha: float) -> np.ndarray:
    """
    alpha > 1 -> higher variance; alpha < 1 -> compressed variance.
    """
    out = data.copy()
    mean = data[:t_shift].mean()
    out[t_shift:] = mean + alpha * (out[t_shift:] - mean)
    return out


def inject_trend_shift(data: np.ndarray, t_shift: int, slope: float) -> np.ndarray:
    """
    Adds a linear drift starting at t_shift.
    slope: value added per time step (e.g. 0.01 * std / step)
    """
    out = data.copy()
    n = len(data) - t_shift
    drift = np.arange(n) * slope
    if out.ndim == 2:
        out[t_shift:] += drift[:, None]
    else:
        out[t_shift:] += drift
    return out


def inject_spike(data: np.ndarray, t_shift: int, magnitude: float) -> np.ndarray:
    """
    Single extreme value at t_shift, data returns to normal after.
    magnitude: k * std
    """
    out = data.copy()
    out[t_shift] += magnitude
    return out


def inject_shift(
    data: np.ndarray,
    shift_type: str,
    t_shift: int,
    magnitude: float,
    train_std: float,
) -> np.ndarray:
    """
    Dispatcher. magnitude is in units of train_std.
    shift_type: 'level' | 'variance' | 'trend' | 'spike'
    """
    m = magnitude * train_std
    if shift_type == "level":
        return inject_level_shift(data, t_shift, m)
    if shift_type == "variance":
        # magnitude here is the alpha multiplier, not offset
        return inject_variance_shift(data, t_shift, alpha=magnitude)
    if shift_type == "trend":
        slope = magnitude * train_std / 100
        return inject_trend_shift(data, t_shift, slope)
    if shift_type == "spike":
        return inject_spike(data, t_shift, m)
    raise ValueError(f"Unknown shift_type: {shift_type}")
