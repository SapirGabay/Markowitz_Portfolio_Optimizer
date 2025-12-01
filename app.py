# =========================================================================
# Markowitz Portfolio Optimizer - Final Code (Optimized for Streamlit Cloud)
# Developed by Sapir Gabay | Industrial Engineering & Intelligent Systems
# =========================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from numpy.linalg import LinAlgError

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
ANNUALIZATION_FACTOR = 12  # Monthly data → annualization


def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculates annualized return (μ) and risk (σ) for the portfolio."""
    weights = np.asarray(weights, dtype=float)

    # Monthly portfolio mean return
    monthly_ret = np.dot(weights, mean_returns.values)
    # Annualized Return (μ)
    returns = monthly_ret * ANNUALIZATION_FACTOR

    # Monthly variance
    monthly_var = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    # Annualized Volatility (σ)
    std_dev = np.sqrt(monthly_var * ANNUALIZATION_FACTOR)

    return std_dev, returns


def minimize_volatility(mean_returns, cov_matrix, constraints, num_assets):
    """Finds the Global Minimum Variance Portfolio (GMVP) using optimization."""
    initial_weights = np.array(num_assets * [1.0 / num_assets], dtype=float)

    # Objective function to minimize (portfolio volatility)
    def objective_function(weights):
        std_dev, _ = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        return std_dev

    # Run the optimization
    result = minimize(
        objective_function,
        initial_weights,
        method="SLSQP",
        bounds=tuple([(0, 1)] * num_assets),  # Weights between 0 and 1
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return result.x


def generate_random_portfolios(mean_returns, cov_matrix, num_assets, num_portfolios=10000):
    """
    Vectorized simulation of many portfolios (for Efficient Frontier).
    Returns array of shape (3, num_portfolios): [std, return, sharpe].
    """
    rng = np.random.default_rng()
    # weights: shape (num_portfolios, num_assets)
    weights = rng.random((num_portfolios, num_assets))
    weights /= weights.sum(axis=1, keepdims=True)

    mean_vec = mean_returns.values  # monthly mean returns
    cov_mat = cov_matrix.values     # monthly covariance

    # Monthly portfolio mean returns (num_portfolios,)
    monthly_returns = weights @ mean_vec
    # Annualized returns
    annual_returns = monthly_returns * ANNUALIZATION_FACTOR

    # Monthly variances via einsum (efficient)
    monthly_vars = np.einsum("ij,jk,ik->i", weights, cov_mat, weights)
    # Annualized std dev
    std_devs = np.sqrt(monthly_vars * ANNUALIZATION_FACTOR)

    # Sharpe ratios (risk-free assumed 0)
    sharpe_ratios = np.divide(
        annual_returns,
        std_devs,
        out=np.zeros_like(annual_returns),
        where=std_devs > 0,
    )

    results = np.vstack((std_devs, annual_returns, sharpe_ratios))
    return results


# =========================================================================
# STREAMLIT UI & APPLICATION LOGIC
# =========================================================================

@st.cache_data
def get_data(tickers, start_date, end_date):
    """Caches yfinance data, ensures monthly resampling, and robust structure handling."""

    # 1. Pull data (Monthly interval)
    raw_data = yf.download(tickers, start=start_date, end=end_date, interval="1mo")

    if raw_data is None or raw_data.empty:
        raise ValueError("No data returned from yfinance. Check tickers and date range.")

    # Handle 'Adj Close' structure (single vs multi tickers)
    if "Adj Close" in raw_data.columns and len(tickers) > 1:
        price_data = raw_data["Adj Close"]
        # Columns expected to be tickers already
    elif "Adj Close" in raw_data.columns and len(tickers) == 1:
        # For single ticker: ensure column name is the ticker, not 'Adj Close'
        price_data = raw_data[["Adj Close"]].copy()
        price_data.columns = tickers
    else:
        raise ValueError("Data retrieval error: Incomplete 'Adj Close' data structure.")

    # 2. Ensure data points are sampled at month-end (ME)
    price_data = price_data.ffill()
    price_data_monthly = price_data.resample("ME").last()

    # Drop columns/rows with all NaNs
    price_data_monthly = price_data_monthly.dropna(axis=1, how="all").dropna(axis=0, how="any")

    return price_data_monthly


# Streamlit page config
st.set_page_config(layout="wide")
st.title("🛡️ Markowitz Portfolio Optimizer: Risk Minimization Model")
st.markdown("---")
st.markdown(
    """
    This model implements the core **Markowitz Portfolio Theory** to calculate the **Efficient Frontier** and determine the 
    **Global Minimum Variance Portfolio (GMVP)** for maximum risk-adjusted returns based on historical data.
    Developed by **Sapir Gabay**, Industrial Engineering & Intelligent Systems.
    """
)
st.markdown("---")

# Sidebar inputs
st.sidebar.header("1. Asset Selection")

ticker_input = st.sidebar.text_area(
    "Enter Stock Tickers, separated by commas (e.g., AAPL, MSFT, GOOG, JPM)",
    "GLD, MSFT",
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
num_assets = len(tickers)

if st.sidebar.button("Run Optimization"):
    # Basic validation
    if num_assets < 2:
        st.error("Please enter at least two stock tickers to perform portfolio optimization.")
    elif start_date >= end_date:
        st.error("Start Date must be earlier than End Date.")
    else:
        try:
            with st.spinner("Pulling historical data and running simulations..."):
                # --- DATA ACQUISITION & PREP ---
                data = get_data(tickers, start_date, end_date)

                if data.empty or data.shape[1] < 2:
                    st.error(
                        "Error: Could not retrieve data for at least two common tickers. "
                        "Try different symbols or check availability."
                    )
                    st.stop()

                actual_tickers = data.columns.tolist()

                returns = data.pct_change().dropna()

                if returns.empty or returns.shape[1] < 2:
                    st.error(
                        "Error: Insufficient common data points for correlation calculation. "
                        "Try a different date range or different tickers."
                    )
                    st.stop()

                # Check for (almost) perfect correlation, ignoring diagonal
                corr = returns.corr().abs()
                np.fill_diagonal(corr.values, 0.0)
                if corr.max().max() >= 0.999:
                    st.error(
                        "Error: Assets are (almost) perfectly correlated. "
                        "Optimization cannot be performed. Choose less correlated assets."
                    )
                    st.stop()

                mean_returns = returns.mean()      # monthly mean returns
                cov_matrix = returns.cov()         # monthly covariance matrix

                # Check covariance matrix is usable (e.g., positive-definite)
                try:
                    # Cholesky is more efficient than full inverse and checks PD
                    np.linalg.cholesky(cov_matrix.values)
                except LinAlgError:
                    st.error(
                        "Error: Covariance Matrix is singular or not positive definite. "
                        "Choose assets with less correlation or a different date range."
                    )
                    st.stop()

                # Constraints: Sum of weights must equal 1
                constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

                # --- OPTIMIZATION & SIMULATION ---

                # 1. Minimum-Risk Portfolio
                weights_min_risk = minimize_volatility(
                    mean_returns, cov_matrix, constraints, len(actual_tickers)
                )

                # Just in case: numeric check
                if not np.allclose(np.sum(weights_min_risk), 1.0, atol=1e-5):
                    raise ValueError("Optimization failed to find valid weights (sum != 1).")

                std_min, ret_min = calculate_portfolio_performance(
                    weights_min_risk, mean_returns, cov_matrix
                )

                # 2. Simulate Efficient Frontier (vectorized)
                results = generate_random_portfolios(
                    mean_returns, cov_matrix, len(actual_tickers), num_portfolios=10000
                )

                # --- PRESENTATION ---
                st.header("Results and Efficient Frontier")

                col1, col2, col3 = st.columns(3)
                col1.metric("Optimal Annualized Return (μ)", f"{ret_min * 100:.2f}%")
                col2.metric("Minimum Annualized Volatility (σ)", f"{std_min * 100:.2f}%")

                sharpe_min = 0.0 if std_min == 0 else ret_min / std_min
                col3.metric("Sharpe Ratio (Min Risk)", f"{sharpe_min:.2f}")

                # --- PLOT EFFICIENT FRONTIER ---
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=results[0, :],
                            y=results[1, :],
                            mode="markers",
                            marker=dict(
                                color=results[2, :],  # Sharpe ratio
                                colorbar=dict(title="Sharpe Ratio"),
                                colorscale="RdYlBu",
                                showscale=True,
                                size=5,
                            ),
                            name="Simulated Portfolios",
                        )
                    ]
                )

                fig.add_trace(
                    go.Scatter(
                        x=[std_min],
                        y=[ret_min],
                        mode="markers",
                        marker=dict(color="green", size=15, symbol="star"),
                        name="Global Minimum Variance Portfolio (GMVP)",
                    )
                )

                fig.update_layout(
                    title="Markowitz Efficient Frontier",
                    xaxis_title="Annualized Volatility (Risk)",
                    yaxis_title="Annualized Return (μ)",
                    hovermode="closest",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- ASSET ALLOCATION TABLE ---
                st.subheader("Optimal Asset Allocation (GMVP)")

                weights_df = pd.DataFrame(
                    {
                        "Asset": actual_tickers,
                        "Optimal Weight": [f"{w * 100:.2f}%" for w in weights_min_risk],
                    }
                )
                st.dataframe(
                    weights_df.sort_values(by="Optimal Weight", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(
                "Optimization Failed. Please change the date range or ticker symbols. "
                f"Error Details: {e}"
            )
