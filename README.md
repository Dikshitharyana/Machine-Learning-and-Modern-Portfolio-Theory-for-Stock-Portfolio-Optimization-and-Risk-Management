# Machine-Learning-and-Modern-Portfolio-Theory-for-Stock-Portfolio-Optimization-and-Risk-Management
ML-Portfolio Optimization
Project Overview
The ML-Portfolio Optimization project focuses on optimizing an investment portfolio using Modern Portfolio Theory (MPT) and machine learning techniques. It leverages historical stock data to calculate optimal asset weights based on expected returns, volatility, and the Sharpe ratio. Additionally, the project uses a Random Forest model to predict future returns, a Monte Carlo simulation for risk management, and data visualizations to analyze the portfolio's performance.

Key Components
Data Collection and Preprocessing

The project uses the yfinance library to collect historical adjusted closing prices for a set of stocks (AAPL, GOOGL, AMZN, MSFT, TSLA).
Data is cleaned and daily returns are calculated.
Portfolio Optimization (Modern Portfolio Theory)

Expected returns, volatility, and covariance matrices are calculated based on historical data.
A Monte Carlo simulation is used to simulate the future performance of the portfolio under different asset weight combinations.
The project identifies the optimal portfolio based on the Sharpe ratio.
Machine Learning Model for Return Prediction

A Random Forest Regressor is trained to predict future returns based on past returns.
Risk Management (Monte Carlo Simulation)

Monte Carlo simulations are used to simulate the potential future value of the portfolio, considering the randomness of returns over time.
Visualizations

Monte Carlo Simulation Graph: Shows the simulation of portfolio value over time.
Efficient Frontier: Displays the tradeoff between risk (volatility) and return, along with the Sharpe ratio.
Moving Averages & Adjusted Close Price: Plots the stock’s adjusted closing price alongside its 50-day and 200-day moving averages.
Installation
Ensure you have the following libraries installed:

bash
Copy
Edit
pip install yfinance pandas numpy matplotlib seaborn scikit-learn scipy plotly
Steps in the Project
Step 1: Data Collection
The data for the project is downloaded using the yfinance library, where the adjusted closing prices for stocks are retrieved for a time period from 2015-01-01 to 2025-01-01.

python
Copy
Edit
assets = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
data = yf.download(assets, start='2015-01-01', end='2025-01-01')['Adj Close']
Step 2: Data Preprocessing
The daily percentage returns are calculated for each asset:

python
Copy
Edit
returns = data.pct_change().dropna()
Step 3: Portfolio Optimization with MPT
Using the Modern Portfolio Theory, we calculate the expected return and volatility of the portfolio. The portfolio optimization function is designed to maximize the Sharpe ratio:

python
Copy
Edit
def portfolio_statistics(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility
Step 4: Random Forest Model for Return Prediction
We train a Random Forest Regressor model to predict future returns:

python
Copy
Edit
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
predictions = model.predict(X)
Step 5: Monte Carlo Simulation
Simulate the future portfolio value based on historical returns using a Monte Carlo method:

python
Copy
Edit
def monte_carlo_simulation(initial_investment, n_simulations, n_days, mean_returns, cov_matrix):
    simulation_results = np.zeros((n_simulations, n_days))
    for i in range(n_simulations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
        portfolio_value = initial_investment
        for day_return in daily_returns:
            portfolio_value *= (1 + np.dot(day_return, optimal_weights))
        simulation_results[i, :] = portfolio_value
    return simulation_results
Step 6: Visualizations
Monte Carlo Simulation Graph: This graph simulates the portfolio value over time for 1000 simulations.

Efficient Frontier Visualization: Shows how portfolios with different weights perform in terms of risk and return.

python
Copy
Edit
plt.figure(figsize=(10, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.show()
Step 7: Results
After optimization, the project outputs the portfolio weights that maximize the Sharpe ratio:

text
Copy
Edit
Maximum Sharpe Ratio Portfolio:
Expected Return: 36.79%
Volatility: 35.97%
  Ticker    Weight
0   AAPL  0.304151
1  GOOGL  0.122832
2   AMZN  0.236545
3   MSFT  0.175417
4   TSLA  0.161055
Conclusion
The project successfully combines machine learning and Modern Portfolio Theory to construct an optimized investment portfolio. It predicts returns using Random Forest, optimizes the portfolio based on the Sharpe ratio, and uses Monte Carlo simulations to model risk. The results can assist investors in making data-driven decisions for their portfolios.

