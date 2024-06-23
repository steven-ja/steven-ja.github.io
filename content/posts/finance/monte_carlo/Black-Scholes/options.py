# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:30:47 2024

@author: stefa
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

def monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, num_steps):
    dt = T / num_steps
    paths = np.zeros((num_simulations, num_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, num_steps + 1):
        z = np.random.standard_normal(num_simulations)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    option_payoffs = np.maximum(paths[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(option_payoffs)
    
    return option_price, paths

# Example usage
S0 = 100  # Initial stock price
K = 98.5   # Strike price
T = 1     # Time to maturity (in years)
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
num_simulations = 10000
num_steps = 252  # Number of trading days in a year

price, paths = monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, num_steps)
print(f"Estimated option price: {price:.3f}")

## Visualization
plt.figure(figsize=(12, 8))
plt.plot(paths[:100, :].T, linewidth=1)
plt.title("Sample Stock Price Paths")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.show()
plt.savefig("images/simulation_path.png", dpi=300)

plt.figure(figsize=(12, 8))
plt.hist(paths[:, -1], bins=100)
plt.title("Distribution of Final Stock Prices")
plt.xlabel("Stock Price")
plt.ylabel("Frequency")
plt.vlines(S0, 0, 350, colors='r', linestyles='--', label=r'$S_0$')
plt.legend()
plt.show()
plt.savefig("images/simulation_histogram.png", dpi=300)


from scipy.stats import norm

def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


bs_price = black_scholes_call(S0, K, T, r, sigma)
print(f"Black-Scholes price: {bs_price:.3f}")
print(f"Monte Carlo price: {price:.3f}")
print(f"Difference: {abs(bs_price - price):.4f}")