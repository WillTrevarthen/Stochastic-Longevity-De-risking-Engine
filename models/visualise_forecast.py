import matplotlib.pyplot as plt
import numpy as np

def plot_kt_forecast(historical_years, historical_kt, forecast_kt):
    """
    historical_years: array of years from your data handler
    historical_kt: the kt values from LC fit
    forecast_kt: the 2D array from LC predict
    """
    n_forecast_years = forecast_kt.shape[0]
    forecast_years = np.arange(historical_years[-1] + 1, historical_years[-1] + 1 + n_forecast_years)
    
    # Calculate Percentiles for the Fan
    lower_bound = np.percentile(forecast_kt, 5, axis=1)
    median = np.percentile(forecast_kt, 50, axis=1)
    upper_bound = np.percentile(forecast_kt, 95, axis=1)

    plt.figure(figsize=(10, 6))
    
    # Plot Historical
    plt.plot(historical_years, historical_kt, color='black', label='Historical κt')
    
    # Plot Forecast Median
    plt.plot(forecast_years, median, color='blue', label='Forecast Median')
    
    # Fill the Fan (90% Confidence Interval)
    plt.fill_between(forecast_years, lower_bound, upper_bound, color='blue', alpha=0.2, label='90% CI')
    
    plt.title('Lee-Carter Longevity Index (κt) Forecast')
    plt.xlabel('Year')
    plt.ylabel('κt')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Run this after your model.predict()
# plot_kt_forecast(handler.matrix.columns, model.kt, results['kt_forecasts'])