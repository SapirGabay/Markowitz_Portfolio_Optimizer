# =========================================================================
# Markowitz Portfolio Optimizer - Final Code (Ready for Streamlit Cloud)
# Developed by Sapir Gabay | Industrial Engineering & Intelligent Systems
# =========================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go

# =========================================================================
# MANDATORY DISCLAIMER (Required for financial applications)
# =========================================================================
# IMPORTANT: This application and the underlying Markowitz model 
# implementation are for **academic and demonstrative purposes only**. 
# 
# This analysis:
# 1. Does not constitute financial advice, investment recommendations, 
#    or an offer to buy or sell any securities.
# 2. Is based solely on historical data and does not guarantee future results.
# 3. Does not account for transaction costs, taxes, or liquidity constraints.
# 
# Users assume all responsibility and risk for any investment decisions made.
# =========================================================================

# =========================================================================
# MARKOWITZ CORE FUNCTIONS (Based on Modern Portfolio Theory)
# =========================================================================

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """ Calculates annualized return (μ) and risk (σ) for the portfolio. """
    
    # Annualized Return (μ)
    returns = np.sum(mean_returns * weights) * 252 
    
    # Annualized Volatility (σ) (The core Markowitz formula)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std_dev, returns

def minimize_volatility(mean_returns, cov_matrix, constraints, num_assets):
    """ Finds the Global Minimum Variance Portfolio (GMVP) using optimization. """
    
    initial_weights = np.array(num_assets*[1./num_assets,])
    
    # Objective function to minimize (portfolio volatility)
    def objective_function(weights):
        return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[0]

    # Run the optimization 
    optimal_weights = minimize(
        objective_function, 
        initial_weights, 
        method='SLSQP', 
        bounds=tuple([(0, 1)] * num_assets), # Weights must be between 0 and 1
        constraints=constraints
    )
    return optimal_weights.x

def generate_random_portfolios(mean_returns, cov_matrix, constraints, num_assets, num_portfolios=10000):
    """ Simulates many portfolios to plot the Efficient Frontier. """
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        std_dev, returns = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0,i] = std_dev
        results[1,i] = returns
        results[2,i] = returns / std_dev # Sharpe Ratio
        
    return results

# =========================================================================
# STREAMLIT UI & APPLICATION LOGIC
# =========================================================================

st.set_page_config(layout="wide")
st.title("🛡️ Markowitz Portfolio Optimizer: Risk Minimization Model")
st.markdown("---")
st.markdown("""
    This model implements the core **Markowitz Portfolio Theory** to calculate the **Efficient Frontier** and determine the 
    **Global Minimum Variance Portfolio (GMVP)** for maximum risk-adjusted returns based on historical data.
    Developed by **Sapir Gabay**, Industrial Engineering & Intelligent Systems.
""")
st.markdown("---")


st.sidebar.header("1. Asset Selection")

# Input for stock tickers
ticker_input = st.sidebar.text_area(
    "Enter Stock Tickers, separated by commas (e.g., AAPL, MSFT, GOOG, JPM)", 
    "AAPL, MSFT, GOOG, JPM"
)

# Input for date range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))


tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
num_assets = len(tickers)

if st.sidebar.button("Run Optimization"):
    if num_assets < 2:
        st.error("Please enter at least two stock tickers to perform portfolio optimization.")
    else:
        try:
            with st.spinner('Pulling historical data and running 10,000 simulations...'):
                
                # --- DATA ACQUISITION & CALCULATION ---
                data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
                
                returns = data.pct_change().dropna()
                mean_returns = returns.mean()
                cov_matrix = returns.cov()
                
                # Constraints: Sum of weights must equal 1
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) 
                
                # --- OPTIMIZATION & SIMULATION ---
                
                # 1. Calculate Minimum Risk Portfolio (Optimization)
                weights_min_risk = minimize_volatility(mean_returns, cov_matrix, constraints, num_assets)
                std_min, ret_min = calculate_portfolio_performance(weights_min_risk, mean_returns, cov_matrix)
                
                # 2. Simulate Random Portfolios (for the Efficient Frontier plot)
                results = generate_random_portfolios(mean_returns, cov_matrix, constraints, num_assets)
                
                
                # --- PRESENTATION ---
                st.header("Results and Efficient Frontier")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Optimal Annualized Return (μ)", f"{ret_min*100:.2f}%")
                col2.metric("Minimum Annualized Volatility (σ)", f"{std_min*100:.2f}%")
                col3.metric("Sharpe Ratio (Min Risk)", f"{ret_min / std_min:.2f}")

                # --- PLOT EFFICIENT FRONTIER ---
                
                # Scatter plot of all simulated portfolios
                fig = go.Figure(data=[
                    go.Scatter(
                        x=results[0,:], 
                        y=results[1,:], 
                        mode='markers',
                        marker=dict(
                            color=results[2,:], # Color based on Sharpe Ratio
                            colorbar=dict(title="Sharpe Ratio"),
                            colorscale='RdYlBu',
                            showscale=True,
                            size=5
                        ),
                        name='Simulated Portfolios'
                    ),
                    # Mark the Minimum Risk Portfolio
                    go.Scatter(
                        x=[std_min], 
                        y=[ret_min], 
                        mode='markers',
                        marker=dict(color='green', size=15, symbol='star'),
                        name='Global Minimum Variance Portfolio (GMVP)'
                    )
                ])

                fig.update_layout(
                    title="Markowitz Efficient Frontier",
                    xaxis_title='Annualized Volatility (Risk)',
                    yaxis_title='Annualized Return (μ)',
                    hovermode="closest",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                
                # --- ASSET ALLOCATION TABLE ---
                
                st.subheader("Optimal Asset Allocation (GMVP)")
                
                weights_df = pd.DataFrame({
                    'Asset': tickers, 
                    'Optimal Weight': [f"{w*100:.2f}%" for w in weights_min_risk]
                })
                st.dataframe(weights_df.sort_values(by='Optimal Weight', ascending=False), use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"An unexpected error occurred during calculation. Please check your ticker symbols or data range. Error: {e}")
