import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import copy
import urllib.request
import json
import hashlib

# =====================================================================
# 1. MODEL ARCHITECTURE BLUEPRINTS (Required for pickle validation)
# =====================================================================

class SabermetricEngine:
    def __init__(self, xwoba_threshold: float = 0.350, exit_velo_threshold: float = 95.0):
        self.xwoba_threshold = xwoba_threshold
        self.exit_velo_threshold = exit_velo_threshold

    def compute_swdecision_plus(self, xwoba: np.ndarray, exit_velo: np.ndarray, out_of_zone: np.ndarray) -> np.ndarray:
        base_value = np.zeros_like(xwoba)
        base_value += np.where(xwoba >= self.xwoba_threshold, 1.25, -0.50)
        ooz_penalty = np.where((out_of_zone == 1) & (exit_velo < self.exit_velo_threshold), -1.5, 0.0)
        ooz_reward = np.where((out_of_zone == 1) & (exit_velo >= self.exit_velo_threshold), 1.0, 0.0)
        return base_value + ooz_penalty + ooz_reward

class BaseNeuralNetwork(nn.Module):
    def __init__(self, input_dim=5, output_classes=2):
        super(BaseNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_classes)
        )
    def forward(self, x):
        return self.network(x)

class IAR2_Refiner:
    def __init__(self, base_classifier: nn.Module, iterations: int = 3, lr: float = 1e-3, batch_size: int = 64):
        self.base_classifier = base_classifier
        self.iterations = iterations
        self.lr = lr
        self.batch_size = batch_size

    def _train_from_scratch(self, X: torch.Tensor, y_soft: torch.Tensor, epochs: int = 15) -> nn.Module:
        model = copy.deepcopy(self.base_classifier)
        return model

    def refine_labels(self, X: np.ndarray, y_noisy: np.ndarray) -> np.ndarray:
        return np.array([[0.5, 0.5]])

def generate_stable_seed(string_input: str, offset: int) -> int:
    sha256_encoded = hashlib.sha256(string_input.encode('utf-8')).hexdigest()
    return int(sha256_encoded[:8], 16) + offset

# =====================================================================
# 2. STREAMLIT INTERACTIVE USER INTERFACE & PORTFOLIO LOGIC
# =====================================================================

st.title("⚾ MLB Quantitative Trading Dashboard")
st.write("Multi-Agent Reinforcement Learning Prediction Engine")

# Bankroll allocation input
bankroll = st.number_input("Enter Daily Starting Bankroll ($):", min_value=0.0, value=1000.0, step=100.0)

# Portfolio Optimization Controls Sidebar
st.sidebar.header("🛡️ Portfolio Optimization Controls")
kelly_fraction = st.sidebar.slider("Kelly Criterion Modifier", 0.10, 1.00, 0.25, step=0.05)
daily_max_exposure_pct = st.sidebar.slider("Max Total Daily Bankroll Exposure (%)", 5.0, 25.0, 10.0, step=1.0)

max_daily_liability = bankroll * (daily_max_exposure_pct / 100.0)

st.write(f"### Current Capital Allocation Baseline: ${bankroll:,.2f}")
st.write(f"⚠️ **Maximum Daily Portfolio Liability Limit:** ${max_daily_liability:,.2f} ({daily_max_exposure_pct}% max exposure)")

# Session state memory configuration to shield display from lag drops
if "trading_slate_calculated" not in st.session_state:
    st.session_state.trading_slate_calculated = False
if "cached_optimized_wagers" not in st.session_state:
    st.session_state.cached_optimized_wagers = []

# The Roster Vault Matrix providing exact real player substitutions for blanks
ROSTER_VAULT = {
    "Cincinnati Reds": {"pitcher": "Hunter Greene", "batter": "Elly De La Cruz"},
    "San Diego Padres": {"pitcher": "Dylan Cease", "batter": "Manny Machado"},
    "New York Yankees": {"pitcher": "Gerrit Cole", "batter": "Aaron Judge"},
    "Los Angeles Dodgers": {"pitcher": "Tyler Glasnow", "batter": "Shohei Ohtani"},
    "Chicago Cubs": {"pitcher": "Shota Imanaga", "batter": "Seiya Suzuki"},
    "Baltimore Orioles": {"pitcher": "Corbin Burnes", "batter": "Gunnar Henderson"},
    "Oakland Athletics": {"pitcher": "JP Sears", "batter": "Brent Rooker"},
    "Boston Red Sox": {"pitcher": "Lucas Giolito", "batter": "Rafael Devers"},
    "San Francisco Giants": {"pitcher": "Logan Webb", "batter": "Matt Chapman"},
    "St. Louis Cardinals": {"pitcher": "Sonny Gray", "batter": "Nolan Arenado"},
    "Cleveland Guardians": {"pitcher": "Tanner Bibee", "batter": "José Ramírez"},
    "Houston Astros": {"pitcher": "Framber Valdez", "batter": "Yordan Alvarez"},
    "Atlanta Braves": {"pitcher": "Spencer Strider", "batter": "Ronald Acuña Jr."},
    "Philadelphia Phillies": {"pitcher": "Zack Wheeler", "batter": "Bryce Harper"},
    "Texas Rangers": {"pitcher": "Jacob deGrom", "batter": "Corey Seager"},
    "Toronto Blue Jays": {"pitcher": "Kevin Gausman", "batter": "Vladimir Guerrero Jr."},
    "Seattle Mariners": {"pitcher": "Luis Castillo", "batter": "Julio Rodríguez"},
    "Miami Marlins": {"pitcher": "Sandy Alcántara", "batter": "Jake Burger"},
    "New York Mets": {"pitcher": "Freddy Peralta", "batter": "Francisco Lindor"},
    "Washington Nationals": {"pitcher": "MacKenzie Gore", "batter": "CJ Abrams"},
    "Tampa Bay Rays": {"pitcher": "Shane Baz", "batter": "Yandy Díaz"},
    "Chicago White Sox": {"pitcher": "Garrett Crochet", "batter": "Luis Robert Jr."},
    "Detroit Tigers": {"pitcher": "Tarik Skubal", "batter": "Riley Greene"},
    "Kansas City Royals": {"pitcher": "Cole Ragans", "batter": "Bobby Witt Jr."},
    "Minnesota Twins": {"pitcher": "Pablo López", "batter": "Byron Buxton"},
    "Colorado Rockies": {"pitcher": "Kyle Freeland", "batter": "Ezequiel Tovar"},
    "Arizona Diamondbacks": {"pitcher": "Zac Gallen", "batter": "Corbin Carroll"},
    "Los Angeles Angels": {"pitcher": "Patrick Sandoval", "batter": "Mike Trout"},
    "Milwaukee Brewers": {"pitcher": "Freddy Peralta", "batter": "William Contreras"},
    "Pittsburgh Pirates": {"pitcher": "Mitch Keller", "batter": "Oneil Cruz"}
}

if st.button("Scan Complete Slate & Optimize Bets"):
    st.info("🔄 Ingesting live game props and normalizing player roster feeds...")
    
    try:
        with open('model.pkl', 'rb') as file:
            model_package = pickle.load(file)
        st.success("✅ Model parameters loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model package: {e}")
        st.stop()
        
    try:
        api_key = st.secrets["THE_ODDS_API_KEY"]
    except Exception:
        api_key = None

    games_found = []
    live_props_extracted = {}
    
    if api_key:
        try:
            schedule_url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h&apiKey={api_key}"
            response = urllib.request.urlopen(schedule_url)
            games_found = json.loads(response.read().decode())
            
            for match in games_found[:3]:
                match_id = match.get('id')
                home_team_name = match.get('home_team')
                try:
                    prop_url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{match_id}/odds?regions=us&markets=player_strikeouts,player_total_bases&oddsFormat=american&apiKey={api_key}"
                    prop_resp = urllib.request.urlopen(prop_url)
                    prop_json = json.loads(prop_resp.read().decode())
                    
                    bookmakers = prop_json.get('bookmakers', [])
                    if bookmakers:
                        for market in bookmakers[0].get('markets', []):
                            market_key = market.get('key')
                            outcomes = market.get('outcomes', [])
                            if outcomes:
                                player_name = outcomes[0].get('description')
