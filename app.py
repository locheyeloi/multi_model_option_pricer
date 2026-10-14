import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import yfinance as yf
import datetime


class OptionPricing:
    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q  

    def d1(self):
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def bs_price(self, is_call: bool = True) -> float:
        """Black-Scholes-Merton (européenne), avec dividende continu q."""
        d1, d2 = self.d1(), self.d2()
        disc_q = np.exp(-self.q * self.T)
        disc_r = np.exp(-self.r * self.T)
        if is_call:
            return self.S * disc_q * si.norm.cdf(d1) - self.K * disc_r * si.norm.cdf(d2)
        return self.K * disc_r * si.norm.cdf(-d2) - self.S * disc_q * si.norm.cdf(-d1)

    def greeks(self, is_call: bool = True) -> dict:
        d1, d2 = self.d1(), self.d2()
        disc_q = np.exp(-self.q * self.T)
        disc_r = np.exp(-self.r * self.T)
        pdf_d1 = si.norm.pdf(d1)

        gamma = disc_q * pdf_d1 / (self.S * self.sigma * np.sqrt(self.T))
        vega = (self.S * disc_q * pdf_d1 * np.sqrt(self.T)) / 100

        if is_call:
            delta = disc_q * si.norm.cdf(d1)
            theta = (-self.S * disc_q * pdf_d1 * self.sigma / (2 * np.sqrt(self.T))
                     - self.r * self.K * disc_r * si.norm.cdf(d2)
                     + self.q * self.S * disc_q * si.norm.cdf(d1)) / 365
            rho = (self.K * self.T * disc_r * si.norm.cdf(d2)) / 100
        else:
            delta = -disc_q * si.norm.cdf(-d1)
            theta = (-self.S * disc_q * pdf_d1 * self.sigma / (2 * np.sqrt(self.T))
                     + self.r * self.K * disc_r * si.norm.cdf(-d2)
                     - self.q * self.S * disc_q * si.norm.cdf(-d1)) / 365
            rho = (-self.K * self.T * disc_r * si.norm.cdf(-d2)) / 100

        return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

    def monte_carlo_pricing(self, is_call: bool = True, simulations: int = 10000, seed=None) -> float:
        """GBM sous probabilité risque-neutre, variables antithétiques, dividende continu q inclus dans le drift."""
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(simulations // 2)

        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * self.T
        diffusion = self.sigma * np.sqrt(self.T)
        ST1 = self.S * np.exp(drift + diffusion * Z)
        ST2 = self.S * np.exp(drift - diffusion * Z)
        ST = np.concatenate((ST1, ST2))

        payoffs = np.maximum(ST - self.K, 0) if is_call else np.maximum(self.K - ST, 0)
        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def binomial_tree_american(self, is_call: bool = True, steps: int = 100) -> float:
        """Arbre binomial CRR avec exercice anticipé à chaque nœud (option américaine)."""
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)

        ST = np.array([self.S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
        V = np.maximum(ST - self.K, 0) if is_call else np.maximum(self.K - ST, 0)

        for i in range(steps - 1, -1, -1):
            ST = ST[:-1] * u   
            continuation = disc * (p * V[1:] + (1 - p) * V[:-1])
            intrinsic = np.maximum(ST - self.K, 0) if is_call else np.maximum(self.K - ST, 0)
            V = np.maximum(continuation, intrinsic)

        return V[0]


st.set_page_config(page_title="Pricer Dérivés Quant", layout="wide")
st.title("Pricer Multi-Modèles d'Options & Grecques")


@st.cache_data
def get_market_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")

        if hist.empty or len(hist) < 2:
            return None, None

        current_price = hist['Close'].iloc[-1]

        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        historical_volatility = log_returns.std() * np.sqrt(252)

        if not np.isfinite(current_price) or not np.isfinite(historical_volatility) or historical_volatility <= 0:
            return None, None

        return float(current_price), float(historical_volatility)
    except Exception:
        return None, None


DEFAULT_S, DEFAULT_SIGMA = 150.0, 0.25 

with st.sidebar:
    st.header("1. Choix de l'Actif")
    ticker_input = st.text_input("Ticker Yahoo Finance (ex: AAPL, TSLA, BNP.PA)", value="AAPL")

    if st.button("Récupérer les données"):
        fetched_S, fetched_sigma = get_market_data(ticker_input)
        if fetched_S is None:
            st.error(f"Impossible de trouver les données pour '{ticker_input}'. Vérifie le ticker.")
        else:
            st.session_state['S'], st.session_state['sigma'] = fetched_S, fetched_sigma

    if 'S' not in st.session_state:
        fetched_S, fetched_sigma = get_market_data(ticker_input)
        if fetched_S is None:
            st.warning(
                f"Impossible de récupérer '{ticker_input}' au chargement — valeurs par défaut utilisées "
                f"(S={DEFAULT_S}, σ={DEFAULT_SIGMA:.0%}). Clique sur « Récupérer les données » pour réessayer."
            )
            fetched_S, fetched_sigma = DEFAULT_S, DEFAULT_SIGMA
        st.session_state['S'], st.session_state['sigma'] = fetched_S, fetched_sigma

    st.divider()

    st.header("2. Paramètres de l'Option")
    option_type = st.radio("Type d'option", ["Call", "Put"], horizontal=True)

    S = st.number_input("Prix actuel du Sous-jacent (S)", value=float(st.session_state['S']), step=1.0)

    default_strike = round(st.session_state['S'] / 5) * 5
    K = st.number_input("Strike (K)", value=float(default_strike), step=1.0)

    T = st.slider("Maturité en années (T)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)

    sigma = st.number_input("Volatilité Historique (σ)", value=float(st.session_state['sigma']), format="%.4f", step=0.01)

    r = st.slider("Taux sans risque (r)", min_value=0.0, max_value=0.1, value=0.05, step=0.001)
    q = st.slider("Taux de dividende continu (q)", min_value=0.0, max_value=0.1, value=0.0, step=0.001)

is_call = (option_type == "Call")
pricer = OptionPricing(S, K, T, r, sigma, q)

st.subheader(f"1. Valorisation de l'Option ({option_type})")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Black-Scholes (Européenne)", f"{pricer.bs_price(is_call):.4f} €")
with col2:
    st.metric("Monte Carlo (10k sim, Antithétique)", f"{pricer.monte_carlo_pricing(is_call):.4f} €")
    st.caption("Varie légèrement à chaque interaction : c'est le bruit statistique normal d'une simulation Monte Carlo.")
with col3:
    st.metric("Arbre Binomial (Américaine)", f"{pricer.binomial_tree_american(is_call):.4f} €")

st.subheader("2. Analyse des Risques (Les Grecques)")
greeks = pricer.greeks(is_call)
g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)

g_col1.metric("Delta (Δ)", f"{greeks['Delta']:,.4f}")
g_col2.metric("Gamma (Γ)", f"{greeks['Gamma']:,.4f}")
g_col3.metric("Theta (Θ) (par jour)", f"{greeks['Theta']:,.4f}")
g_col4.metric("Vega (ν) (par 1%)", f"{greeks['Vega']:,.4f}")
g_col5.metric("Rho (ρ) (par 1%)", f"{greeks['Rho']:,.4f}")

st.subheader(f"3. Surface de Prix Black-Scholes ({option_type} vs Spot & Temps)")


@st.cache_data
def generate_3d_surface(K, r, q, sigma, is_call):
    spot_range = np.linspace(max(1, K * 0.5), K * 1.5, 50)
    time_range = np.linspace(0.01, 2, 50)
    spot_grid, time_grid = np.meshgrid(spot_range, time_range)

    d1 = (np.log(spot_grid / K) + (r - q + 0.5 * sigma ** 2) * time_grid) / (sigma * np.sqrt(time_grid))
    d2 = d1 - sigma * np.sqrt(time_grid)
    disc_q = np.exp(-q * time_grid)
    disc_r = np.exp(-r * time_grid)

    if is_call:
        prices = spot_grid * disc_q * si.norm.cdf(d1) - K * disc_r * si.norm.cdf(d2)
    else:
        prices = K * disc_r * si.norm.cdf(-d2) - spot_grid * disc_q * si.norm.cdf(-d1)

    return spot_grid, time_grid, prices


X, Y, Z = generate_3d_surface(K, r, q, sigma, is_call)

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
fig.update_layout(
    title=f'Prix du {option_type} (Axe Z) selon le Spot (Axe X) et la Maturité (Axe Y)',
    autosize=False,
    width=900, height=600,
    scene=dict(
        xaxis_title='Prix du Sous-jacent (Spot)',
        yaxis_title='Temps à Maturité (Années)',
        zaxis_title=f'Prix du {option_type}'
    )
)
st.plotly_chart(fig, use_container_width=True)
