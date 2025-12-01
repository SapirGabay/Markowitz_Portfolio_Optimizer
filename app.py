# =========================================================================
# Markowitz Portfolio Optimizer - Streamlit App (GMVP + Max Sharpe + Equal Weight)
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
# DISCLAIMER
# =========================================================================
# This application is for academic and demonstrative purposes only.
# It does NOT constitute financial advice or investment recommendations.
# =========================================================================

ANNUALIZATION_FACTOR = 12  # Monthly → Annual


# ======================= CORE MARKOWITZ FUNCTIONS ========================

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate annualized portfolio volatility (std_dev) and return (returns).
    mean_returns and cov_matrix are monthly; we annualize.
    """
    weights = np.asarray(weights, dtype=float)

    # Monthly portfolio mean return
    monthly_ret = np.dot(weights, mean_returns.values)
    # Annualized return
    returns = monthly_ret * ANNUALIZATION_FACTOR

    # Monthly variance
    monthly_var = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    # Annualized std dev
    std_dev = np.sqrt(monthly_var * ANNUALIZATION_FACTOR)

    return std_dev, returns


def minimize_volatility(mean_returns, cov_matrix, constraints, num_assets):
    """Find Global Minimum Variance Portfolio (GMVP) via SLSQP."""

    initial_weights = np.array(num_assets * [1.0 / num_assets], dtype=float)

    def objective_function(weights):
        std_dev, _ = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        return std_dev

    result = minimize(
        objective_function,
        initial_weights,
        method="SLSQP",
        bounds=tuple([(0, 1)] * num_assets),  # long-only, 0–100%
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Optimization failed (GMVP): {result.message}")

    return result.x


def maximize_sharpe(mean_returns, cov_matrix, constraints, num_assets, risk_free_rate):
    """Find Maximum Sharpe Ratio portfolio (long-only)."""

    initial_weights = np.array(num_assets * [1.0 / num_assets], dtype=float)

    def negative_sharpe(weights):
        std_dev, ret = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        # Annualized Sharpe with annual risk-free rate
        if std_dev == 0:
            return 1e6
        sharpe = (ret - risk_free_rate) / std_dev
        return -sharpe  # minimize negative Sharpe → maximize Sharpe

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=tuple([(0, 1)] * num_assets),
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Optimization failed (Max Sharpe): {result.message}")

    return result.x


def generate_random_portfolios(mean_returns, cov_matrix, num_assets, num_portfolios=10000):
    """
    Vectorized random portfolio simulation.
    Returns array: [std_devs, annual_returns, sharpe_ratios]
    """
    rng = np.random.default_rng()
    weights = rng.random((num_portfolios, num_assets))
    weights /= weights.sum(axis=1, keepdims=True)

    mean_vec = mean_returns.values  # monthly means
    cov_mat = cov_matrix.values     # monthly covariance

    # Monthly mean returns for each portfolio
    monthly_returns = weights @ mean_vec
    annual_returns = monthly_returns * ANNUALIZATION_FACTOR

    # Monthly variances via einsum
    monthly_vars = np.einsum("ij,jk,ik->i", weights, cov_mat, weights)
    std_devs = np.sqrt(monthly_vars * ANNUALIZATION_FACTOR)

    # Sharpe (rf = 0) עבור סימולציות
    sharpe_ratios = np.divide(
        annual_returns,
        std_devs,
        out=np.zeros_like(annual_returns),
        where=std_devs > 0,
    )

    return np.vstack((std_devs, annual_returns, sharpe_ratios))


# =========================== DATA HANDLING ===============================

@st.cache_data
def get_data(tickers, start_date, end_date):
    """
    Download monthly price data from yfinance, resample to month-end,
    and clean NaNs.
    """
    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )

    if raw_data is None or raw_data.empty:
        raise ValueError("No data returned from yfinance. Check tickers and date range.")

    # Multi-ticker: MultiIndex columns ('Adj Close', TICKER)
    if "Adj Close" in raw_data.columns and len(tickers) > 1:
        price_data = raw_data["Adj Close"]
    # Single ticker: make sure column is the ticker name
    elif "Adj Close" in raw_data.columns and len(tickers) == 1:
        price_data = raw_data[["Adj Close"]].copy()
        price_data.columns = tickers
    else:
        raise ValueError("Data retrieval error: 'Adj Close' not found in downloaded data.")

    # Forward-fill, then resample to month-end
    price_data = price_data.ffill()
    price_data_monthly = price_data.resample("ME").last()

    # Drop empty columns and rows with any NaNs
    price_data_monthly = price_data_monthly.dropna(axis=1, how="all").dropna(axis=0, how="any")

    return price_data_monthly


# ============================ STREAMLIT UI ===============================

st.set_page_config(layout="wide")

st.title("🛡️ Markowitz Portfolio Optimizer")
st.markdown("---")
st.markdown(
    """
    This app implements core **Markowitz Portfolio Theory** to compute the **Efficient Frontier**
    and three key portfolios:

    - **GMVP** – Global Minimum Variance Portfolio (minimum risk)
    - **Max Sharpe** – Portfolio with the highest risk-adjusted return
    - **Equal Weight** – Simple 1/N benchmark

    Developed by **Sapir Gabay** | Industrial Engineering & Management student.
    """
)
st.markdown(
    """
    **How to use:**
    1. Enter **2–10 tickers** (stocks / ETFs) separated by commas (e.g., `GLD, QQQ, TLT`).
    2. Choose a **date range** (3–5 years is usually a good starting point).
    3. Set the **risk-free annual rate** for the Sharpe ratio (e.g., `0.02` = 2% per year), or keep the default.
    4. Click **Run Optimization** to see the Efficient Frontier and the three portfolios.
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

st.sidebar.header("2. Sharpe Settings")
risk_free_rate = st.sidebar.number_input(
    "Risk-free annual rate (for Sharpe)",
    min_value=0.0,
    max_value=0.20,
    value=0.02,
    step=0.005,
    format="%.3f",
)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
num_assets = len(tickers)

if st.sidebar.button("Run Optimization"):
    # Basic validations
    if num_assets < 2:
        st.error("Please enter at least two stock tickers to perform portfolio optimization.")
    elif start_date >= end_date:
        st.error("Start Date must be earlier than End Date.")
    else:
        try:
            with st.spinner("Pulling historical data and running simulations..."):
                # ----- DATA -----
                data = get_data(tickers, start_date, end_date)

                if data.empty or data.shape[1] < 2:
                    st.error(
                        "Error: Could not retrieve data for at least two common tickers. "
                        "Try different symbols or a different date range."
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

                # Correlation check (ignore diagonal)
                corr = returns.corr().abs()
                np.fill_diagonal(corr.values, 0.0)
                if corr.max().max() >= 0.999:
                    st.error(
                        "Error: Assets are (almost) perfectly correlated. "
                        "Choose less correlated assets or change the date range."
                    )
                    st.stop()

                mean_returns = returns.mean()   # monthly means
                cov_matrix = returns.cov()      # monthly covariance

                # Covariance matrix sanity check (positive-definite)
                try:
                    np.linalg.cholesky(cov_matrix.values)
                except LinAlgError:
                    st.error(
                        "Error: Covariance matrix is singular or not positive definite. "
                        "Choose assets with less correlation or a different date range."
                    )
                    st.stop()

                # Sum of weights must be 1
                constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

                # ----- OPTIMIZATION -----
                # GMVP
                weights_gmvp = minimize_volatility(
                    mean_returns, cov_matrix, constraints, len(actual_tickers)
                )
                if not np.allclose(np.sum(weights_gmvp), 1.0, atol=1e-5):
                    raise ValueError("GMVP: weights do not sum to 1.")

                std_gmvp, ret_gmvp = calculate_portfolio_performance(
                    weights_gmvp, mean_returns, cov_matrix
                )

                # Equal Weight portfolio
                weights_eq = np.array(len(actual_tickers) * [1.0 / len(actual_tickers)])
                std_eq, ret_eq = calculate_portfolio_performance(
                    weights_eq, mean_returns, cov_matrix
                )

                # Max Sharpe portfolio
                weights_ms = maximize_sharpe(
                    mean_returns, cov_matrix, constraints, len(actual_tickers), risk_free_rate
                )
                if not np.allclose(np.sum(weights_ms), 1.0, atol=1e-5):
                    raise ValueError("Max Sharpe: weights do not sum to 1.")

                std_ms, ret_ms = calculate_portfolio_performance(
                    weights_ms, mean_returns, cov_matrix
                )

                sharpe_gmvp = 0.0 if std_gmvp == 0 else (ret_gmvp - risk_free_rate) / std_gmvp
                sharpe_eq = 0.0 if std_eq == 0 else (ret_eq - risk_free_rate) / std_eq
                sharpe_ms = 0.0 if std_ms == 0 else (ret_ms - risk_free_rate) / std_ms

                # ----- SIMULATIONS -----
                results = generate_random_portfolios(
                    mean_returns, cov_matrix, len(actual_tickers), num_portfolios=10000
                )

                # ----- OUTPUT -----
                st.header("Results and Efficient Frontier")

                col1, col2, col3 = st.columns(3)
                col1.metric("GMVP Annualized Return (μ)", f"{ret_gmvp * 100:.2f}%")
                col2.metric("GMVP Annualized Volatility (σ)", f"{std_gmvp * 100:.2f}%")
                col3.metric("GMVP Sharpe (rf = {:.1f}%)".format(risk_free_rate * 100),
                            f"{sharpe_gmvp:.2f}")

                # Efficient Frontier plot
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=results[0, :],
                            y=results[1, :],
                            mode="markers",
                            marker=dict(
                                color=results[2, :],
                                colorbar=dict(title="Sharpe Ratio (rf=0)"),
                                colorscale="RdYlBu",
                                showscale=True,
                                size=5,
                            ),
                            name="Simulated Portfolios",
                        )
                    ]
                )

                # GMVP
                fig.add_trace(
                    go.Scatter(
                        x=[std_gmvp],
                        y=[ret_gmvp],
                        mode="markers",
                        marker=dict(color="green", size=14, symbol="star"),
                        name="GMVP (Min Volatility)",
                    )
                )

                # Max Sharpe
                fig.add_trace(
                    go.Scatter(
                        x=[std_ms],
                        y=[ret_ms],
                        mode="markers",
                        marker=dict(color="orange", size=12, symbol="triangle-up"),
                        name="Max Sharpe",
                    )
                )

                # Equal Weight
                fig.add_trace(
                    go.Scatter(
                        x=[std_eq],
                        y=[ret_eq],
                        mode="markers",
                        marker=dict(color="black", size=10, symbol="x"),
                        name="Equal Weight",
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

                # ----- COMPARISON TABLE -----
                st.subheader("Portfolio Comparison")
                comparison_df = pd.DataFrame(
                    {
                        "Portfolio": ["GMVP (Min Vol)", "Max Sharpe", "Equal Weight"],
                        "Annual Return (μ)": [
                            f"{ret_gmvp * 100:.2f}%",
                            f"{ret_ms * 100:.2f}%",
                            f"{ret_eq * 100:.2f}%",
                        ],
                        "Annual Volatility (σ)": [
                            f"{std_gmvp * 100:.2f}%",
                            f"{std_ms * 100:.2f}%",
                            f"{std_eq * 100:.2f}%",
                        ],
                        f"Sharpe (rf={risk_free_rate*100:.1f}%)": [
                            f"{sharpe_gmvp:.2f}",
                            f"{sharpe_ms:.2f}",
                            f"{sharpe_eq:.2f}",
                        ],
                    }
                )
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # ----- ASSET ALLOCATIONS -----
                st.subheader("Optimal Asset Allocations")

                st.markdown("**GMVP (Global Minimum Variance Portfolio)**")
                gmvp_df = pd.DataFrame(
                    {
                        "Asset": actual_tickers,
                        "Weight": [f"{w * 100:.2f}%" for w in weights_gmvp],
                    }
                )
                st.dataframe(
                    gmvp_df.sort_values(by="Weight", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("**Max Sharpe Portfolio**")
                ms_df = pd.DataFrame(
                    {
                        "Asset": actual_tickers,
                        "Weight": [f"{w * 100:.2f}%" for w in weights_ms],
                    }
                )
                st.dataframe(
                    ms_df.sort_values(by="Weight", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("**Equal Weight Portfolio**")
                eq_df = pd.DataFrame(
                    {
                        "Asset": actual_tickers,
                        "Weight": [f"{w * 100:.2f}%" for w in weights_eq],
                    }
                )
                st.dataframe(
                    eq_df,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(
                "Optimization Failed. Please change the date range or ticker symbols. "
                f"Error Details: {e}"
            )
