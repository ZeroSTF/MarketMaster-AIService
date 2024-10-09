import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils.data_fetcher import get_stock_data

def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def optimize(portfolio, risk_free_rate=0.01):
    data = pd.DataFrame()
    for symbol in portfolio:
        stock_data = get_stock_data(symbol)
        data[symbol] = stock_data['Close']
    
    returns = data.pct_change()
    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(portfolio)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    std, ret = portfolio_annualized_performance(result.x, mean_returns, cov_matrix)
    sharpe_ratio = (ret - risk_free_rate) / std
    
    return {
        'weights': dict(zip(portfolio, result.x)),
        'expected_annual_return': ret,
        'annual_volatility': std,
        'sharpe_ratio': sharpe_ratio
    }