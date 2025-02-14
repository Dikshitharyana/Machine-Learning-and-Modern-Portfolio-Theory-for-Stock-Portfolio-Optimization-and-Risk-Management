# Machine-Learning-and-Modern-Portfolio-Theory-for-Stock-Portfolio-Optimization-and-Risk-Management
ML-Portfolio Optimization
Project Overview
ML-Portfolio Optimization is a Python project aimed at optimizing investment portfolios using Modern Portfolio Theory (MPT) combined with machine learning techniques. The project uses historical stock data to compute optimal asset weights based on expected returns, volatility, and the Sharpe ratio. It incorporates machine learning to predict future returns and uses Monte Carlo simulations for risk management.

Table of Contents
Installation
Project Structure
Key Components
Data Collection and Preprocessing
Portfolio Optimization
Machine Learning Model for Return Prediction
Risk Management (Monte Carlo Simulation)
Visualizations
Running the Project
Results
Conclusion
Installation
To run this project, make sure you have the following Python libraries installed:

bash
Copy
Edit
pip install yfinance pandas numpy matplotlib seaborn scikit-learn scipy plotly
Project Structure
main.py: The main script for portfolio optimization, machine learning, and visualizations.
requirements.txt: Lists the required dependencies.
README.md: This file, providing a project overview and instructions.
Key Components
Data Collection and Preprocessing
The data is fetched from Yahoo Finance using yfinance. Stock data is downloaded for multiple assets (AAPL, GOOGL, AMZN, MSFT, TSLA) from the specified date range.

python
Copy
Edit
assets = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']
data = yf.download(assets, start='2015-01-01', end='2025-01-01')['Adj Close']
Daily returns are calculated using percentage change:

python
Copy
Edit
returns = data.pct_change().dropna()
Portfolio Optimization (Modern Portfolio Theory)
Using Modern Portfolio Theory (MPT), we calculate portfolio statistics such as expected returns, volatility, and covariance. The project uses Monte Carlo simulations to identify the optimal portfolio that maximizes the Sharpe ratio.

python
Copy
Edit
def portfolio_statistics(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility
Machine Learning Model for Return Prediction
We train a Random Forest Regressor model to predict future stock returns based on historical returns.

python
Copy
Edit
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
predictions = model.predict(X)
Risk Management (Monte Carlo Simulation)
The Monte Carlo simulation is used to model the future portfolio value based on random simulations of returns over time.

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
Visualizations
Monte Carlo Simulation Graph: Shows the distribution of the portfolio value over time based on the simulations.
Efficient Frontier Visualization: Displays portfolios that optimize the risk-return tradeoff.
Moving Averages & Adjusted Close Price: Visualizes stock price data along with 50-day and 200-day moving averages.
python
Copy
Edit
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
Running the Project
Clone the repository or download the project files.
Install the required dependencies from requirements.txt.
Run the main Python script (main.py) to execute the portfolio optimization, machine learning, and visualizations.
bash
Copy
Edit
python main.py
Results
The output of the optimization process includes the expected return, volatility, and portfolio weights for each asset in the portfolio.

Maximum Sharpe Ratio Portfolio
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
The ML-Portfolio Optimization project successfully combines Modern Portfolio Theory (MPT) and machine learning to construct an optimized investment portfolio. By predicting returns with a Random Forest model and optimizing asset weights for maximum Sharpe ratio, the project provides investors with a comprehensive strategy for portfolio management. The Monte Carlo simulations offer a robust risk management technique, and the visualizations enhance decision-making.
