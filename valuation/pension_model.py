import numpy as np

class PensionLiabilityModel:
    def __init__(self, initial_age, initial_count, annual_pension):
        self.initial_age = initial_age
        self.initial_count = initial_count
        self.annual_pension = annual_pension

    def calculate_survival_paths(self, mx_forecasts, start_age_idx):
        """
        mx_forecasts: 3D array (Age, Year, Simulation) from LeeCarter
        start_age_idx: The index in the matrix corresponding to initial_age
        """
        n_ages, n_years, n_sims = mx_forecasts.shape
        
        # We follow the cohort until they hit the max age in our data
        projection_horizon = min(n_years, n_ages - start_age_idx - 1)
        
        # Survival probabilities (n_years, n_sims)
        # k_p_x: Probability of surviving from start to year k
        k_p_x = np.ones((projection_horizon + 1, n_sims))
        
        for s in range(n_sims):
            cumulative_prob = 1.0
            for t in range(projection_horizon):
                # Diagonal lookup: Age increases with Year
                m_xt = mx_forecasts[start_age_idx + t, t, s]
                
                # Convert central death rate to annual prob of death
                q_xt = 1 - np.exp(-m_xt)
                
                cumulative_prob *= (1 - q_xt)
                k_p_x[t+1, s] = cumulative_prob
                
        return k_p_x

    def project_cashflows(self, k_p_x):
        # Expected survivors * Pension Amount
        return self.initial_count * k_p_x * self.annual_pension
    
    def present_value(self, cashflows, spot_rates):
        """
        cashflows: (Years, Sims)
        spot_rates: 1D array of discount rates for each year (e.g., 0.04 for 4%)
        """
        n_years = cashflows.shape[0]
        # Create discount factors: 1 / (1 + r)^t
        t = np.arange(n_years)
        discount_factors = 1 / (1 + spot_rates)**t
        
        # Multiply each year's cashflow across all sims by the discount factor
        pv_per_sim = np.sum(cashflows * discount_factors[:, np.newaxis], axis=0)
        
        return pv_per_sim