import numpy as np
import matplotlib.pyplot as plt

def calculate_risk_metrics(unhedged_pvs, hedged_pvs):
    """
    Calculates VaR and Expected Shortfall at 95%
    """
    metrics = {}
    for name, data in [("Unhedged", unhedged_pvs), ("Hedged", hedged_pvs)]:
        # Value-at-Risk (95th percentile of cost)
        var_95 = np.percentile(data, 95)
        # Expected Shortfall (Average of the worst 5% of cases)
        es_95 = data[data >= var_95].mean()
        
        metrics[name] = {"VaR_95": var_95, "ES_95": es_95, "StdDev": np.std(data)}
    
    return metrics

def plot_hedging_effectiveness(unhedged_pvs, hedged_pvs):
    plt.figure(figsize=(12, 6))
    
    # Histogram of total costs
    plt.hist(unhedged_pvs, bins=50, alpha=0.5, label='Unhedged (Pure Liability)', color='red')
    plt.hist(hedged_pvs, bins=50, alpha=0.5, label='Hedged (Liability + Swap)', color='green')
    
    plt.axvline(np.mean(unhedged_pvs), color='red', linestyle='dashed', linewidth=2, label='Mean Unhedged')
    plt.axvline(np.mean(hedged_pvs), color='green', linestyle='dashed', linewidth=2, label='Mean Hedged')

    plt.title('Risk Reduction: Longevity Swap Effectiveness')
    plt.xlabel('Present Value of Total Cost (GBP)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()