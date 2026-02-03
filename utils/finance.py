import numpy as np

def get_flat_yield_curve(rate, length):
    """Simple starting point: a flat 4% curve."""
    return np.full(length, rate)

def get_real_yield_curve_example(length):
    """
    Simulates an upward sloping yield curve: 
    Starts at 3%, rises to 5% over 30 years.
    """
    return 0.03 + (0.02 * (np.arange(length) / 30))