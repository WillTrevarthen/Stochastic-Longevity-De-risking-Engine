import matplotlib.pyplot as plt
import seaborn as sns

def plot_mortality_surface(log_matrix):
    plt.figure(figsize=(12, 8))
    # Heatmap of log-mortality
    sns.heatmap(log_matrix, cmap='viridis_r')
    plt.title('Log-Mortality Surface (Age vs. Year)')
    plt.xlabel('Year')
    plt.ylabel('Age')
    plt.show()

    # Cross-section: Mortality at specific ages over time
    log_matrix.loc[[30, 50, 70, 85]].T.plot(figsize=(10, 6))
    plt.title('Log-Mortality Trends for Specific Ages')
    plt.ylabel('ln(mx)')
    plt.grid(True)
    plt.show()