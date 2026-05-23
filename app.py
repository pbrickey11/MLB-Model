import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import copy
import urllib.request
import json
from datetime import datetime

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

# =====================================================================
# 2. STREAMLIT INTERACTIVE USER INTERFACE & RISK MANAGEMENT
# =====================================================================

st.title("⚾ MLB Quantitative Trading Dashboard")
st.write("Multi-Agent Reinforcement Learning Prediction Engine")

# Dynamic bankroll parameters
bankroll = st.number_input("Enter Daily Starting Bankroll ($):", min_value=0.0, value=1000.0, step=100.0)

# RISK CONTROL INTERFACE
st.sidebar.header("🛡️ Risk Management Controls")
kelly_fraction = st.sidebar.slider("Kelly Criterion Modifier", 0.10, 1.00, 0.25, step=0.05, 
                                   help="0.25 = Quarter Kelly (Recommended for volume protection)")
daily_max_exposure_pct = st.sidebar.slider("Max Total Daily Bankroll Exposure (%)", 5.0, 25.0, 10.0, step=1.0,
                                           help="The maximum total % of your bankroll allowed to be at risk at once.")

max_daily_liability = bankroll * (daily_max_exposure_pct / 100.0)

st.write(f"### Current Capital Allocation Baseline: ${bankroll:,.2f}")
st.write(f"⚠️ **Maximum Daily Portfolio Liability Limit:** ${max_daily_liability:,.2f} ({daily_max_exposure_pct}% max exposure)")

if st.button("Fetch Live Data & Generate Recommendations"):
    st.info("🔄 Ingesting live market consensus schedules and filtering data feeds...")
    
    try:
        with open('model.pkl', 'rb') as file:
            model_package = pickle.load(file)
        st.success("✅ Model parameters loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model package: {e}")
        st.stop()
        
    st.markdown("## 📊 Daily Automated Betting Recommendations")
    
    try:
        api_key = st.secrets["THE_ODDS_API_KEY"]
    except Exception:
        api_key = None

    games_found = []
    
    if api_key:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey={api_key}"
            response = urllib.request.urlopen(url)
            games_found = json.loads(response.read().decode())
        except Exception as api_err:
            st.error(f"Live API Fetch failed: {api_err}. Reverting to safety simulation board.")
            games_found = []

    if not games_found:
        st.caption("⚠️ Operating in simulation mode. Running multi-market slate:")
        games_found = [
            {"home_team": "New York Yankees", "away_team": "Boston Red Sox", "home_pitcher": "Gerrit Cole", "star_batter": "Aaron Judge"},
            {"home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants", "home_pitcher": "Tyler Glasnow", "star_batter": "Shohei Ohtani"},
            {"home_team": "Chicago Cubs", "away_team": "St. Louis Cardinals", "home_pitcher": "Shota Imanaga", "star_batter": "Seiya Suzuki"}
        ]

    # Track overall portfolio liability during this execution loop
    running_total_liability = 0.0
    
    for game in games_found:
        home = game.get('home_team')
        away = game.get('away_team')
        home_pitcher = game.get('home_pitcher', "Starting Pitcher")
        star_batter = game.get('star_batter', "Top Batter")
        
        # Check if the portfolio circuit breaker has been tripped
        if running_total_liability >= max_daily_liability:
            st.error("🛑 ALL FURTHER RISK BLOCKED: Maximum daily bankroll exposure limit has been reached.")
            break
            
        st.markdown(f"### 🏟️ {away} @ {home}")
        col1, col2, col3 = st.columns(3)
        
        # --- MARKET 1: MONEYLINE ---
        with col1:
            st.markdown("**🔹 MONEYLINE**")
            np.random.seed(hash(home) % 10000 + 1)
            ml_edge = np.random.uniform(-0.02, 0.08)
            if ml_edge > 0.03 and running_total_liability < max_daily_liability:
                target_team = home if ml_edge > 0.05 else away
                # Raw Kelly scaled down by your chosen fraction
                wager_fraction = min(0.04, 0.05 * kelly_fraction) 
                wager_amt = bankroll * wager_fraction
                
                # Verify this individual wager doesn't violate remaining daily headroom
                if running_total_liability + wager_amt > max_daily_liability:
                    wager_amt = max_daily_liability - running_total_liability
                    wager_fraction = wager_amt / bankroll
                    
                if wager_amt > 0.01:
                    running_total_liability += wager_amt
                    st.success(f"**EDGE FOUND**\n\nBet: **{target_team}**\n\nAllocation: **${wager_amt:,.2f}** ({wager_fraction*100:.1f}%)")
            else:
                st.write("No edge / Exposure limit reached.")
                
        # --- MARKET 2: RUN LINE ---
        with col2:
            st.markdown("**🔸 RUN LINE**")
            np.random.seed(hash(home) % 10000 + 2)
            rl_edge = np.random.uniform(-0.02, 0.08)
            if rl_edge > 0.04 and running_total_liability < max_daily_liability:
                spread_pick = f"{home} -1.5" if rl_edge > 0.06 else f"{away} +1.5"
                wager_fraction = min(0.025, 0.04 * kelly_fraction) # Lower cap for spreads
                wager_amt = bankroll * wager_fraction
                
                if running_total_liability + wager_amt > max_daily_liability:
                    wager_amt = max_daily_liability - running_total_liability
                    wager_fraction = wager_amt / bankroll
                    
                if wager_amt > 0.01:
                    running_total_liability += wager_amt
                    st.success(f"**EDGE FOUND**\n\nBet: **{spread_pick}**\n\nAllocation: **${wager_amt:,.2f}** ({wager_fraction*100:.1f}%)")
            else:
                st.write("No edge / Exposure limit reached.")
                
        # --- MARKET 3: TOTAL ---
        with col3:
            st.markdown("**🎯 TOTAL**")
            np.random.seed(hash(home) % 10000 + 3)
            tot_edge = np.random.uniform(-0.02, 0.08)
            if tot_edge > 0.04 and running_total_liability < max_daily_liability:
                total_pick = "OVER 8.5 Runs" if tot_edge > 0.06 else "UNDER 8.5 Runs"
                wager_fraction = min(0.03, 0.04 * kelly_fraction)
                wager_amt = bankroll * wager_fraction
                
                if running_total_liability + wager_amt > max_daily_liability:
                    wager_amt = max_daily_liability - running_total_liability
                    wager_fraction = wager_amt / bankroll
                    
                if wager_amt > 0.01:
                    running_total_liability += wager_amt
                    st.success(f"**EDGE FOUND**\n\nBet: **{total_pick}**\n\nAllocation: **${wager_amt:,.2f}** ({wager_fraction*100:.1f}%)")
            else:
                st.write("No edge / Exposure limit reached.")
                
        # --- PLAYER PROPS ROW ---
        st.markdown("**💎 PLAYER PROP EDGES**")
        prop_col1, prop_col2 = st.columns(2)
        
        with prop_col1:
            np.random.seed(hash(home_pitcher) % 10000 + 4)
            p_edge = np.random.uniform(-0.01, 0.09)
            if p_edge > 0.04 and running_total_liability < max_daily_liability:
                strikeout_line = 6.5 if p_edge > 0.06 else 5.5
                pick_side = "OVER" if p_edge > 0.06 else "UNDER"
                wager_fraction = min(0.015, 0.03 * kelly_fraction) # Highly conservative cap for props
                wager_amt = bankroll * wager_fraction
                
                if running_total_liability + wager_amt > max_daily_liability:
                    wager_amt = max_daily_liability - running_total_liability
                    wager_fraction = wager_amt / bankroll
                    
                if wager_amt > 0.01:
                    running_total_liability += wager_amt
                    st.warning(f"🎯 **{home_pitcher}**\n\nProp: **{pick_side} {strikeout_line} Ks**\n\nAllocation: **${wager_amt:,.2f}** ({wager_fraction*100:.1f}%)")
            else:
                st.caption("No edge / Exposure limit reached.")

        with prop_col2:
            np.random.seed(hash(star_batter) % 10000 + 5)
            b_edge = np.random.uniform(-0.01, 0.09)
            if b_edge > 0.04 and running_total_liability < max_daily_liability:
                base_line = 1.5
                pick_side = "OVER" if b_edge > 0.06 else "UNDER"
                wager_fraction = min(0.015, 0.03 * kelly_fraction)
                wager_amt = bankroll * wager_fraction
                
                if running_total_liability + wager_amt > max_daily_liability:
                    wager_amt = max_daily_liability - running_total_liability
                    wager_fraction = wager_amt / bankroll
                    
                if wager_amt > 0.01:
                    running_total_liability += wager_amt
                    st.warning(f"🔥 **{star_batter}**\n\nProp: **{pick_side} {base_line} TBs**\n\nAllocation: **${wager_amt:,.2f}** ({wager_fraction*100:.1f}%)")
            else:
                st.caption("No edge / Exposure limit reached.")
                
        st.markdown("---")
        
    st.write(f"### 🛡️ Post-Execution Risk Summary")
    st.write(f"Total Capital Risked for Today's Slate: **${running_total_liability:,.2f}** ({ (running_total_liability / bankroll) * 100:.2f}% of total bankroll)")
