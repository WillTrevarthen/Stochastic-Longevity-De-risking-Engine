# Stochastic Longevity & De-risking Engine (SLIDE)

**SLIDE** is a production-grade actuarial engine designed to model, value, and hedge longevity risk in Defined Benefit (DB) pension schemes. It combines demographic time-series forecasting (Data Science), stochastic mortality modeling (Actuarial Maths), and risk-neutral derivative pricing (Quant Finance).

## 🚀 Project Overview

Pension funds face "Longevity Risk"—the financial threat that members live longer than predicted, leading to unfunded liabilities. This engine allows a fund to:

1.  **Forecast** future mortality rates using the stochastic Lee-Carter framework.
2.  **Value** liabilities by projecting cohort cashflows through thousands of simulated futures.
3.  **Hedge** the risk by pricing a **Longevity Swap** using a risk-neutral Wang Transform.

---

## 🛠️ Technical Architecture

### 1. Data Ingestion & Wrangling

The engine processes raw data from the **Human Mortality Database (HMD)**.

- **Matrix Construction:** Transforms long-form death rates ($m_{x,t}$) into an Age-Period-Cohort matrix.
- **Log-Space Transformation:** Operates in log-mortality space to linearize exponential mortality trends (Gompertz Law).

### 2. Stochastic Modeling (Lee-Carter)

We implement the Lee-Carter model, the industry standard for mortality forecasting:
$$\ln(m_{x,t}) = a_x + b_x \kappa_t + \epsilon_{x,t}$$

- **Decomposition:** Parameters are extracted using **Singular Value Decomposition (SVD)**.
- **Forecasting:** The mortality index ($\kappa_t$) is modeled as a **Random Walk with Drift**:
  $$\kappa_t = \kappa_{t-1} + \theta + \xi_t, \quad \xi_t \sim \mathcal{N}(0, \sigma^2)$$

### 3. Quant Finance: Longevity Swap Pricing

To price the hedge, we move from the real-world measure ($P$) to the risk-neutral measure ($Q$) using the **Wang Transform**:
$$\Phi^*(p) = \Phi(\Phi^{-1}(p) + \lambda)$$

- **Risk Premium:** A $\lambda$ parameter (market price of risk) is applied to survival probabilities.
- **Swap Structure:** The engine calculates the **Fixed Leg** (guaranteed premium) vs. the **Floating Leg** (realized payments).

---

## 📊 Results & Risk Reduction

The engine demonstrates a near-perfect hedge when the swap notional matches the liability.

| Metric                 | Unhedged Portfolio        | Hedged Portfolio         |
| :--------------------- | :------------------------ | :----------------------- |
| **Mean Cost**          | £194.3M (Best Estimate)   | £194.5M (Risk-Adjusted)  |
| **Standard Deviation** | ~£117,000                 | **~£0.00**               |
| **Risk Profile**       | Volatile, Tail-Risk Heavy | Deterministic, Immunized |

---

## 💻 Installation & Usage

### Prerequisites

- Python 3.9+
- NumPy, Pandas, SciPy, Matplotlib, Seaborn

### Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/longevity-engine.git](https://github.com/yourusername/longevity-engine.git)
   cd longevity-engine
   ```
