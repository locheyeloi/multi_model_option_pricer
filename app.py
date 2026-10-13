import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import yfinance as yf
import datetime

class OptionPricing:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def bs_call(self):
        return (self.S * si.norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0))
    
    def bs_put(self):
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.S * si.norm.cdf(-self.d1(), 0.0, 1.0))

    def greeks(self):
        d1 = self.d1()
        d2 = self.d2()
        
        delta_call = si.norm.cdf(d1)
        gamma = si.norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        theta_call = (-self.S * si.norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2)) / 365
        vega = (self.S * si.norm.pdf(d1) * np.sqrt(self.T)) / 100
        rho_call = (self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(d2)) / 100
        
        return {"Delta": delta_call, "Gamma": gamma, "Theta": theta_call, "Vega": vega, "Rho": rho_call}

    def monte_carlo_pricing(self, simulations=10000):
        Z = np.random.standard_normal(simulations // 2)
        
        ST1 = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        ST2 = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T - self.sigma * np.sqrt(self.T) * Z)
        
        ST = np.concatenate((ST1, ST2))
        payoffs = np.maximum(ST - self.K, 0)
        
        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def binomial_tree_american_call(self, steps=100):
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        ST = np.array([self.S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
        V = np.maximum(ST - self.K, 0)
        
        for i in range(steps - 1, -1, -1):
            ST = ST[:-1] / u
            V = np.maximum(np.exp(-self.r * dt) * (p * V[1:] + (1 - p) * V[:-1]), ST - self.K)
            
        return V[0]

st.set_page_config(page_title="Pricer Dérivés Quant", layout="wide")
st.title("📈 Pricer Multi-Modèles d'Options & Grecques")

@st.cache_data
def get_market_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return None, None
            
        current_price = hist['Close'].iloc[-1]
        
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        historical_volatility = log_returns.std() * np.sqrt(252)
        
        return current_price, historical_volatility
    except Exception as e:
        return None, None

with st.sidebar:
    st.header("1. Choix de l'Actif")
    ticker_input = st.text_input("Ticker Yahoo Finance (ex: AAPL, TSLA, BNP.PA)", value="AAPL")
    
    if st.button("Récupérer les données"):
        st.session_state['S'], st.session_state['sigma'] = get_market_data(ticker_input)
        if st.session_state['S'] is None:
            st.error(f"Impossible de trouver les données pour {ticker_input}")

    if 'S' not in st.session_state:
        st.session_state['S'], st.session_state['sigma'] = get_market_data(ticker_input)

    st.divider()
    
    st.header("2. Paramètres de l'Option")
    S = st.number_input("Prix actuel du Sous-jacent (S)", value=float(st.session_state['S']), step=1.0)
    
    default_strike = round(st.session_state['S'] / 5) * 5
    K = st.number_input("Strike (K)", value=float(default_strike), step=1.0)
    
    T = st.slider("Maturité en années (T)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
    
    sigma = st.number_input("Volatilité Historique (σ)", value=float(st.session_state['sigma']), format="%.4f", step=0.01)
    
    r = st.slider("Taux sans risque (r)", min_value=0.0, max_value=0.1, value=0.05, step=0.001)

pricer = OptionPricing(S, K, T, r, sigma)

st.subheader("1. Valorisation de l'Option (Call)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Black-Scholes (Européenne)", f"{pricer.bs_call():.4f} €")
with col2:
    st.metric("Monte Carlo (10k sim, Antithétique)", f"{pricer.monte_carlo_pricing():.4f} €")
with col3:
    st.metric("Arbre Binomial (Américaine)", f"{pricer.binomial_tree_american_call():.4f} €")

st.subheader("2. Analyse des Risques (Les Grecques)")
greeks = pricer.greeks()
g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)

g_col1.metric("Delta (Δ)", f"{greeks['Delta']:,.4f}")
g_col2.metric("Gamma (Γ)", f"{greeks['Gamma']:,.4f}")
g_col3.metric("Theta (Θ) (par jour)", f"{greeks['Theta']:,.4f}")
g_col4.metric("Vega (ν) (par 1%)", f"{greeks['Vega']:,.4f}")
g_col5.metric("Rho (ρ) (par 1%)", f"{greeks['Rho']:,.4f}")

st.subheader("3. Surface de Prix Black-Scholes (Prix vs Spot & Temps)")

@st.cache_data
def generate_3d_surface(K, r, sigma):
    spot_range = np.linspace(max(1, K*0.5), K*1.5, 50)
    time_range = np.linspace(0.01, 2, 50)
    spot_grid, time_grid = np.meshgrid(spot_range, time_range)
    
    d1 = (np.log(spot_grid / K) + (r + 0.5 * sigma**2) * time_grid) / (sigma * np.sqrt(time_grid))
    d2 = d1 - sigma * np.sqrt(time_grid)
    call_prices = (spot_grid * si.norm.cdf(d1) - K * np.exp(-r * time_grid) * si.norm.cdf(d2))
    
    return spot_grid, time_grid, call_prices

X, Y, Z = generate_3d_surface(K, r, sigma)

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
fig.update_layout(
    title='Prix du Call (Axe Z) selon le Spot (Axe X) et la Maturité (Axe Y)',
    autosize=False,
    width=900, height=600,
    scene=dict(
        xaxis_title='Prix du Sous-jacent (Spot)',
        yaxis_title='Temps à Maturité (Années)',
        zaxis_title='Prix du Call'
    )
)
st.plotly_chart(fig, use_container_width=True)
