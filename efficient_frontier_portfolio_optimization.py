# import libraries
from scipy.optimize import minimize, linprog
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import pandas_datareader as web
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


# get the data
# tickers from my screener 
tickers = pd.read_csv('D:/portfolio.csv').Symbol.tolist()
tickers.remove('XXLLD')
# tickers = ['GOOG', 'SBUX', 'KSS', 'NEM']

# ------------------------------------------------------------------------------
# Getting the close prices for all stocks
start_date, end_date = datetime(2024, 1, 1), datetime.now().date()
stock_prices = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date
).Close

stock_prices.dropna(axis=1, inplace=True)

stock_prices.plot(figsize=(10, 8))
plt.grid()
plt.tight_layout()

# return
returns = stock_prices.pct_change().iloc[1:] * 100

# mean return
mean_return = returns.mean()
# covariance
cov = returns.cov()
cov_np = cov.to_numpy()

N = 10000
D = len(mean_return)
returns = np.zeros(N)
risks = np.zeros(N)
random_weights = []
for i in range(N):
  rand_range = 1.0
  w = np.random.random(D)*rand_range - rand_range / 2 # with short-selling
  w[-1] = 1 - w[:-1].sum()
  np.random.shuffle(w)
  random_weights.append(w)
  ret = mean_return.dot(w)
  risk = np.sqrt(w.dot(cov_np).dot(w))
  returns[i] = ret
  risks[i] = risk
  
plt.figure(figsize=(10,8))
plt.scatter(risks, returns, alpha=0.1)
plt.xlabel('Risk', fontsize=14)
plt.ylabel('Return', fontsize=14)
plt.tight_layout()
# ------------------------------------------------------------------------------
# using Linear Programming to get min and max returns

D = len(mean_return)  # number of assets

A_eq = np.ones((1, D))
b_eq = np.ones(1)

### NOTE: The bounds are by default (0, None) unless otherwise specified.
# bounds = None
bounds = [(0, None)]*D
bounds

# minimize to get minimum return
min_res = linprog(mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
min_return = min_res.fun

# maximize to get maximum return
# maximize
max_res = linprog(-mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
max_return = - max_res.fun


# ------------------------------------------------------------------------------
# Mean - Variance optimal portflolios
# quadratic programming [QP] to minimze Variance

# 100 possible returns between min and max 
N = 100
target_returns = np.linspace(min_return, max_return, num=N)

# object 
def get_portfolio_variance(weights):
  return weights.dot(cov).dot(weights)

# constraints
def target_return_constraint(weights, target):
  return weights.dot(mean_return) - target

def portfolio_constraint(weights):
  return weights.sum() - 1

constraints = [
    {
        'type': 'eq',
        'fun': target_return_constraint,
        'args': [target_returns[0]], # will be updated in loop
    },
    {
        'type': 'eq',
        'fun': portfolio_constraint,
    }
]

optimized_risks = []
for target in target_returns:
  # set target return constraint
  constraints[0]['args'] = [target]

  res = minimize(
      fun=get_portfolio_variance,
      x0=np.ones(D) / D, # uniform
      method='SLSQP',
      constraints=constraints,
      bounds=bounds,
  )
  optimized_risks.append(np.sqrt(res.fun))
  if res.status != 0:
    print(res)
        
plt.figure(figsize=(10,8))
plt.title('Portfolio Optimization Using Quadrant Programming',
          fontsize= 14)
plt.scatter(risks, returns, alpha=0.1);
plt.plot(optimized_risks, target_returns, c='black');
plt.xlabel('Volatility', fontsize=14)
plt.ylabel('Return', fontsize=14)
plt.grid()
plt.tight_layout()
            
    
    
    
    
    
    
    
    
    
    
    
    
    
















