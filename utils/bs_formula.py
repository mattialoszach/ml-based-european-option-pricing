import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculates Black-Scholes-Price of an european call/put option.
    For more information: https://en.wikipedia.org/wiki/Black–Scholes_model

    Parameters:
        S (float): Price of underlying asset at t=0
        K (float): Strike Price
        T (float): Time of option expiration (in years)
        r (float): Annualized risk-free interest rate
        sigma (float): Volatility
        option_type (str): 'call' or 'put'

    Returns:
        float: option price
    """
    # Edge case to avoid division by 0 or NaN
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return price

# Optional: Expected values for sanity check
if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    call_price = black_scholes_price(S, K, T, r, sigma, option_type='call')
    put_price = black_scholes_price(S, K, T, r, sigma, option_type='put')
    
    print(f"Call Price: {call_price:.4f}")
    print(f"Put Price: {put_price:.4f}")
    
    # It can be tested that:
    # Call ≈ 10.45, Put ≈ 5.57
    assert abs(call_price - 10.45) < 0.2, "Call price out of expected range!"
    assert abs(put_price - 5.57) < 0.2, "Put price out of expected range!"
    print("All tests passed.")