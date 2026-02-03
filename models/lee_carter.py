import numpy as np
import pandas as pd

class LeeCarterModel:
    def __init__(self):
        self.ax = None
        self.bx = None
        self.kt = None
        self.ages = None
        self.years = None

    def fit(self, log_mx_matrix):
        """
        Fits the Lee-Carter model: ln(m_xt) = ax + bx*kt
        """
        self.ages = log_mx_matrix.index
        self.years = log_mx_matrix.columns
        
        # 1. ax is the mean of log-mortality across time for each age
        self.ax = log_mx_matrix.mean(axis=1).values
        
        # 2. Center the matrix
        centered = log_mx_matrix.subtract(log_mx_matrix.mean(axis=1), axis=0)
        
        # 3. Singular Value Decomposition (SVD)
        # We want the first principal component
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        
        # bx: Age sensitivity (First column of U)
        # kt: Time trend (First row of Vh * first singular value)
        bx_raw = U[:, 0]
        kt_raw = Vh[0, :] * S[0]
        
        # 4. Normalization (Standard Actuarial Constraint)
        # sum(bx) = 1 and sum(kt) = 0 (centering takes care of sum(kt)=0)
        bx_sum = np.sum(bx_raw)
        self.bx = bx_raw / bx_sum
        self.kt = kt_raw * bx_sum
        
        return self
    
    def predict(self, years_to_forecast, n_sims=1000):
        """
        Forecasts kt and generates stochastic mortality rates (mx).
        Returns: A dictionary containing 'kt_forecasts' and 'mx_forecasts'.
        """
        # 1. Estimate Drift (d) and Volatility (sigma) from historical kt
        dk = np.diff(self.kt)
        drift = np.mean(dk)
        sigma = np.std(dk)
        
        # 2. Simulate future kt paths
        # Shape: (number of years, number of simulations)
        kt_forecasts = np.zeros((years_to_forecast, n_sims))
        last_kt = self.kt[-1]
        
        for s in range(n_sims):
            innovations = np.random.normal(drift, sigma, years_to_forecast)
            kt_forecasts[:, s] = last_kt + np.cumsum(innovations)
            
        # 3. Project mx for each simulation
        # ln(m_xt) = ax + bx * kt
        # mx = exp(ax + bx * kt)
        # We need to reshape ax and bx for broadcasting: (Ages, 1) and (Ages, 1)
        ax_reshaped = self.ax[:, np.newaxis]
        bx_reshaped = self.bx[:, np.newaxis]
        
        # This creates a 3D array: (Age, Year, Simulation)
        log_mx_forecasts = ax_reshaped[:, :, np.newaxis] + \
                           (bx_reshaped[:, :, np.newaxis] * kt_forecasts[np.newaxis, :, :])
        
        mx_forecasts = np.exp(log_mx_forecasts)
        
        return {
            'kt_forecasts': kt_forecasts,
            'mx_forecasts': mx_forecasts
        }