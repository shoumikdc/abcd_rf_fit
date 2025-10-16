import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import cho_factor, cho_solve
import warnings


def complex_fit(f, xdata, ydata, p0=None, weights=None, calc_pcov_mode="cho", **kwargs):
    """
    Wrapper around scipy least_square for complex functions

    Args:
        f: The model function, f(x, …). It must take the independent variable
            as the first argument and the parameters to fit as separate
            remaining arguments.
        xdata: The independent variable where the data is measured.
        ydata: The dependent data.
        p0: Initial guess on independent variables.
        weights: Optional weighting in the calculation of the cost function.
        calc_pcov_mode: Method to use for calculating the covariance matrix.
            Options are 'cho' (Cholesky decomposition), 'svd' (Singular Value Decomposition),
            or 'inv' (direct matrix inversion). 

            ▸ 'cho' is generally fastest and most accurate for well-conditioned problems.
            ▸ 'svd' is more robust for poorly conditioned problems. A bit slower than cho, 
               and small numerical differences are expected. 
            ▸ 'inv' can be unstable for ill-conditioned problems and the slowest. Never recommended.
            
            Default is 'cho', with automatic fallback to 'svd' if Cholesky fails.

        kwargs: passed to the leas_square function  

    Returns:
        A tuple with the optimal parameters and the covariance matrix
    """

    if (np.array(ydata).size - len(p0)) <= 0:
        raise ValueError(
            "yData length should be greater than the number of parameters."
        )

    def residuals(params, x, y):
        """Computes the residual for the least square algorithm"""
        if weights is not None:
            diff = weights * (f(x, *params) - y) # Weights should multiply both terms here
        else:
            diff = f(x, *params) - y
        flat_diff = np.zeros(diff.size * 2, dtype=np.float64)
        flat_diff[0 : flat_diff.size : 2] = diff.real
        flat_diff[1 : flat_diff.size : 2] = diff.imag
        return flat_diff

    kwargs_ls = kwargs.copy()
    kwargs_ls.setdefault("max_nfev", 1000)
    kwargs_ls.setdefault("ftol", 1e-2)
    opt_res = least_squares(residuals, p0, args=(xdata, ydata), **kwargs_ls)

    popt = opt_res.x
    jac = opt_res.jac
    m = opt_res.fun.size
    n = opt_res.x.size
    if m <= n:
        pcov = np.full((n, n), np.nan)
        warnings.warn(
            "Insufficient data to estimate covariance matrix: number of residuals (m=%d) "
            "is not greater than number of parameters (n=%d). Returning NaN covariance."
            % (m, n),
            RuntimeWarning,
        )

        return popt, pcov
        
    if calc_pcov_mode == "cho":
        try:
            JTJ = jac.T @ jac
            c, lower = cho_factor(JTJ, check_finite=False)
            pcov = cho_solve((c, lower), np.eye(JTJ.shape[0]), check_finite=False)
        except Exception:
            warnings.warn(
                "Cholesky decomposition failed; falling back to 'svd' mode to compute covariance.",
                RuntimeWarning,
            )
            calc_pcov_mode = "svd"

    if calc_pcov_mode == "svd":
        _, s, VT = np.linalg.svd(jac, full_matrices=False)
        V = VT.T
        pcov = (V / (s**2)) @ V.T

    if calc_pcov_mode == "inv":
        JTJ = jac.T @ jac
        pcov = np.linalg.inv(JTJ)

    # In all cases, scale covariance
    σ2 = 2.0 * opt_res.cost / (m - n)
    pcov *= σ2
       
    return popt, pcov  
    
    # From abcd_rf_fit (old, can delete later)
    popt = opt_res.x
    jac = opt_res.jac
    cost = opt_res.cost
    pcov = np.linalg.inv(jac.T.dot(jac))
    pcov *= cost / (np.array(ydata).size - len(p0))
    return popt, pcov


def guess_edelay_from_gradient(freq, signal, n=-1):
    """Estimate electrical delay from phase gradient across frequency.

    This function estimates the electrical delay by computing the mean
    phase difference between the beginning and end of the frequency sweep.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array in Hz.
    signal : np.ndarray
        Complex signal array.
    n : int, optional
        Number of points to use from each end (default: -1, uses all points).

    Returns
    -------
    float
        Estimated electrical delay in seconds.
    """
    dtheta = np.mean(np.angle(signal[-n:] / zeros2eps(signal[:n])))
    df = np.mean(np.diff(freq))

    return dtheta / df / 2 / np.pi


def smooth_gradient(signal):
    """Compute smoothed gradient of a signal using Gaussian derivative kernel.

    This function applies a Gaussian derivative convolution to compute
    a smoothed version of the signal gradient, which is used for
    weighting in the ABCD fitting algorithm.

    Parameters
    ----------
    signal : np.ndarray
        Input complex signal array.

    Returns
    -------
    np.ndarray
        Smoothed gradient of the input signal.
    """

    def dnormaldx(x, x_0, sigma):
        return -(x - x_0) * np.exp(-0.5 * ((x - x_0) / sigma) ** 2)

    conv_kernel_size = max(min(100, signal.size // 20), 2)

    conv_kernel = dnormaldx(
        x=np.arange(0.5, conv_kernel_size + 0.5, 1),
        x_0=conv_kernel_size / 2,
        sigma=conv_kernel_size / 8,
    )

    gradient = np.convolve(signal, conv_kernel, "same")
    gradient[: conv_kernel_size // 2] = gradient[
        conv_kernel_size // 2 : 2 * (conv_kernel_size // 2)
    ][::-1]
    gradient[-(conv_kernel_size // 2) :] = gradient[
        -2 * (conv_kernel_size // 2) : -(conv_kernel_size // 2)
    ][::-1]

    return gradient


eps = np.finfo(float).eps


def zeros2eps(x):
    """Replace zeros with machine epsilon to avoid division by zero.

    This utility function replaces any zero values in the input with
    the machine epsilon to prevent numerical issues in calculations.

    Parameters
    ----------
    x : float, complex, or np.ndarray
        Input value or array.

    Returns
    -------
    np.ndarray
        Array with zeros replaced by machine epsilon.
    """
    y = np.array(x)
    y[np.abs(y) < eps] = eps

    return y


def dB(x):
    """Convert magnitude to decibels.

    Parameters
    ----------
    x : float, complex, or np.ndarray
        Input value(s) to convert.

    Returns
    -------
    float or np.ndarray
        Magnitude in decibels (20*log10(|x|)).
    """
    return 20 * np.log10(np.abs(x))


def deg(x):
    """Convert phase from radians to degrees.

    Parameters
    ----------
    x : float, complex, or np.ndarray
        Input complex value(s).

    Returns
    -------
    float or np.ndarray
        Phase in degrees.
    """
    return np.angle(x) * 180 / np.pi


def get_prefix(x):

    prefix = [
        "y",  # yocto
        "z",  # zepto
        "a",  # atto
        "f",  # femto
        "p",  # pico
        "n",  # nano
        "u",  # micro
        "m",  # mili
        "",
        "k",  # kilo
        "M",  # mega
        "G",  # giga
        "T",  # tera
        "P",  # peta
        "E",  # exa
        "Z",  # zetta
        "Y",  # yotta
    ]

    max_x = np.abs(np.max(x))

    if max_x > 10 * eps:

        index = int(np.log10(max_x) / 3 + 8)
        return (x * 10 ** (-3 * (index - 8)), prefix[index])

    else:

        return (0, "")


def get_prefix_str(x, precision: int = 2) -> str:
    """Format a value with appropriate SI prefix as a string.

    Combines the functionality of get_prefix() with string formatting
    to produce a human-readable representation of a value with SI units.

    Parameters
    ----------
    x : float or np.ndarray
        Input value to format.
    precision : int, optional
        Number of decimal places (default: 2).

    Returns
    -------
    str
        Formatted string with value and SI prefix.

    Examples
    --------
    >>> get_prefix_str(1500)
    '1.50 k'
    >>> get_prefix_str(0.001)
    '1.00 m'
    """
    return f"%.{precision}f %s" % get_prefix(x)
