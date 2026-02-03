import numpy as np
from scipy.stats import norm

class LongevitySwapPricer:
    def __init__(self, lambda_parameter=0.25):
        """
        lambda_parameter: The 'market price of risk'. 
        Higher means the insurer charges more for the risk.
        """
        self.lam = lambda_parameter

    def wang_transform(self, survival_probs):
        """
        Transforms real-world survival probabilities into risk-neutral ones.
        """
        # Ensure probs are not exactly 0 or 1 to avoid inf in norm.ppf
        p = np.clip(survival_probs, 1e-10, 1 - 1e-10)
        transformed_p = norm.cdf(norm.ppf(p) + self.lam)
        return transformed_p

    def price_legs(self, real_world_survival_paths, spot_rates):
        """
        real_world_survival_paths: (Years, Sims) from Step 3
        spot_rates: Yield curve for discounting
        """
        n_years, n_sims = real_world_survival_paths.shape
        t = np.arange(n_years)
        discount_factors = 1 / (1 + spot_rates)**t

        # 1. Floating Leg: The actual realized survival (per simulation)
        # PV = sum over time of (Actual Survival * Discount)
        floating_leg_pvs = np.sum(
            real_world_survival_paths * discount_factors[:, np.newaxis], 
            axis=0
        )

        # 2. Fixed Leg (The Price):
        # We take the BEST ESTIMATE survival (mean) and apply the Wang Transform
        best_estimate_survival = np.mean(real_world_survival_paths, axis=1)
        risk_neutral_survival = self.wang_transform(best_estimate_survival)
        
        fixed_leg_pv = np.sum(risk_neutral_survival * discount_factors)

        return {
            'fixed_leg_pv': fixed_leg_pv,
            'floating_leg_pvs': floating_leg_pvs,
            'net_swap_value': floating_leg_pvs - fixed_leg_pv
        }
    
    def price_legs(self, real_world_survival_paths, spot_rates, initial_count=1, annual_pension=1):
        """
        Added initial_count and annual_pension to scale the swap to the actual liability.
        """
        n_years, n_sims = real_world_survival_paths.shape
        t = np.arange(n_years)
        discount_factors = 1 / (1 + spot_rates)**t
        
        # Scale factor
        notional = initial_count * annual_pension

        # 1. Floating Leg (Scaled to GBP)
        floating_leg_pvs = np.sum(
            real_world_survival_paths * discount_factors[:, np.newaxis], 
            axis=0
        ) * notional

        # 2. Fixed Leg (Scaled to GBP)
        best_estimate_survival = np.mean(real_world_survival_paths, axis=1)
        risk_neutral_survival = self.wang_transform(best_estimate_survival)
        
        fixed_leg_pv = np.sum(risk_neutral_survival * discount_factors) * notional

        return {
            'fixed_leg_pv': fixed_leg_pv,
            'floating_leg_pvs': floating_leg_pvs,
            'net_swap_value': floating_leg_pvs - fixed_leg_pv
        }