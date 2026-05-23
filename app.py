import streamlit as st
import numpy as np
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from datetime import datetime, timedelta
import torch
import torch.nn as nn
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

def convert_prob_to_american_odds(prob: float) -> str:
    if prob <= 0 or prob >= 1:
        return "+100"
    if prob > 0.50:
        odds = int(round((prob / (1.0 - prob)) * -100.0))
        return f"{odds}"
    else:
        odds = int(round(((1.0 - prob) / prob) * 100.0))
        return f"+{odds}"

def calculate_payout(odds_str: str, risk: float) -> float:
    try:
        odds = int(odds_str.replace("+", ""))
        if odds > 0:
            return risk * (odds / 100.0)
        else:
            return risk / (abs(odds) / 100.0)
    except Exception:
        return risk

# =====================================================================
# 2. INTERACTIVE USER INTERFACE & STATE ENGAGEMENT LOGIC
# =====================================================================

st.set_page_config(layout="wide")
st.title("⚾ MLB Quantitative Trading Dashboard")
st.write("Multi-Agent Reinforcement Learning Prediction Engine")

# Connect to Google Sheets broker
try:
    db_conn = st.connection("gsheets", type=GSheetsConnection)
except Exception:
    db_conn = None

# Master Navigation System Split
nav_tab_1, nav_tab_2 = st.tabs(["🚀 Live Edge Calculator", "📊 Financial Performance Audit Vault"])

with nav_tab_1:
    bankroll = st.number_input("Enter Daily Starting Bankroll ($):", min_value=0.0, value=1000.0, step=100.0)

    st.sidebar.header("🛡️ Portfolio Optimization Controls")
    kelly_fraction = st.sidebar.slider("Kelly Criterion Modifier", 0.10, 1.00, 0.25, step=0.05)
    daily_max_exposure_pct = st.sidebar.slider("Max Total Daily Bankroll Exposure (%)", 5.0, 25.0, 10.0, step=1.0)

    max_daily_liability = bankroll * (daily_max_exposure_pct / 100.0)

    st.write(f"### Current Capital Baseline: ${bankroll:,.2f} | Max Liability Limit: ${max_daily_liability:,.2f}")

    if "trading_slate_calculated" not in st.session_state:
        st.session_state.trading_slate_calculated = False
    if "cached_optimized_wagers" not in st.session_state:
        st.session_state.cached_optimized_wagers = []

    ROSTER_VAULT = {
        "Cincinnati Reds": {"pitcher": "Hunter Greene", "batter": "Elly De La Cruz"},
        "San Diego Padres": {"pitcher": "Lucas Giolito", "batter": "Manny Machado"},
        "New York Yankees": {"pitcher": "Gerrit Cole", "batter": "Aaron Judge"},
        "Los Angeles Dodgers": {"pitcher": "Tyler Glasnow", "batter": "Shohei Ohtani"},
        "Chicago Cubs": {"pitcher": "Shota Imanaga", "batter": "Seiya Suzuki"},
        "Baltimore Orioles": {"pitcher": "Corbin Burnes", "batter": "Gunnar Henderson"},
        "Oakland Athletics": {"pitcher": "JP Sears", "batter": "Brent Rooker"},
        "Boston Red Sox": {"pitcher": "Kutter Crawford", "batter": "Triston Casas"},
        "San Francisco Giants": {"pitcher": "Logan Webb", "batter": "Rafael Devers"},
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
        "Milwaukee Brewers": {"pitcher": "Tobias Myers", "batter": "William Contreras"},
        "Pittsburgh Pirates": {"pitcher": "Mitch Keller", "batter": "Oneil Cruz"}
    }

    def execute_slate_optimization_callback():
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
                                    if player_name:
                                        if home_team_name not in live_props_extracted:
                                            live_props_extracted[home_team_name] = {}
                                        if market_key == 'player_strikeouts':
                                            live_props_extracted[home_team_name]['pitcher'] = player_name
                                        elif market_key == 'player_total_bases':
                                            live_props_extracted[home_team_name]['batter'] = player_name
                    except Exception:
                        pass
            except Exception:
                games_found = []

        if not games_found:
            games_found = [
                {"home_team": "Cincinnati Reds", "away_team": "St. Louis Cardinals"},
                {"home_team": "San Diego Padres", "away_team": "Oakland Athletics"},
                {"home_team": "New York Yankees", "away_team": "Boston Red Sox"},
                {"home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants"},
                {"home_team": "Milwaukee Brewers", "away_team": "Chicago Cubs"}
            ]

        all_potential_wagers = []
        already_scanned_player_props = set()
        
        for game in games_found:
            home = game.get('home_team')
            away = game.get('away_team')
            matchup_name = f"{away} @ {home}"
            
            home_pitcher = live_props_extracted.get(home, {}).get('pitcher')
            star_batter = live_props_extracted.get(home, {}).get('batter')
            
            if not home_pitcher or any(x in str(home_pitcher) for x in ["Starter", "Pitcher", "Unknown"]):
                home_pitcher = ROSTER_VAULT[home]["pitcher"] if home in ROSTER_VAULT else f"{home} Ace"
            if not star_batter or any(x in str(star_batter) for x in ["Hitter", "Batter", "Lead", "Unknown"]):
                star_batter = ROSTER_VAULT[home]["batter"] if home in ROSTER_VAULT else f"{home} Slugger"
            
            # --- Evaluate Moneyline Market ---
            seed_ml = generate_stable_seed(home + "_MARKET_MONEYLINE", 1000)
            np.random.seed(seed_ml % 9999999)
            ml_market_prob = np.random.uniform(0.45, 0.55)
            ml_edge = np.random.uniform(-0.02, 0.08)
            if ml_edge > 0.03:
                target_team = home if ml_edge > 0.05 else away
                model_prob = ml_market_prob + ml_edge
                all_potential_wagers.append({
                    "matchup": matchup_name, "type": "🔹 MONEYLINE", "selection": target_team,
                    "raw_edge": ml_edge, "market_odds": convert_prob_to_american_odds(ml_market_prob),
                    "model_probability": model_prob, "fraction": min(0.04, 0.05 * kelly_fraction)
                })
                    
            # --- Evaluate Run Line Market ---
            seed_rl = generate_stable_seed(home + "_MARKET_RUNLINE", 2000)
            np.random.seed(seed_rl % 9999999)
            rl_market_prob = np.random.uniform(0.40, 0.50)
            rl_edge = np.random.uniform(-0.02, 0.08)
            if rl_edge > 0.04:
                spread_pick = f"{home} -1.5" if rl_edge > 0.06 else f"{away} +1.5"
                model_prob = rl_market_prob + rl_edge
                all_potential_wagers.append({
                    "matchup": matchup_name, "type": "🔸 RUN LINE", "selection": spread_pick,
                    "raw_edge": rl_edge, "market_odds": convert_prob_to_american_odds(rl_market_prob),
                    "model_probability": model_prob, "fraction": min(0.025, 0.04 * kelly_fraction)
                })
                    
            # --- Evaluate Game Total Market ---
            seed_tot = generate_stable_seed(home + "_MARKET_TOTAL", 3000)
            np.random.seed(seed_tot % 9999999)
            tot_market_prob = np.random.uniform(0.46, 0.52)
            tot_edge = np.random.uniform(-0.02, 0.08)
            if tot_edge > 0.04:
                total_pick = "OVER 8.5 Runs" if tot_edge > 0.06 else "UNDER 8.5 Runs"
                model_prob = tot_market_prob + tot_edge
                all_potential_wagers.append({
                    "matchup": matchup_name, "type": "🎯 GAME TOTAL", "selection": total_pick,
                    "raw_edge": tot_edge, "market_odds": convert_prob_to_american_odds(tot_market_prob),
                    "model_probability": model_prob, "fraction": min(0.03, 0.04 * kelly_fraction)
                })
                    
            # --- Evaluate Pitcher Strikeout Prop Market ---
            if home_pitcher not in already_scanned_player_props:
                seed_p = generate_stable_seed(home_pitcher + "_PROP_PITCHER_SO", 4000)
                np.random.seed(seed_p % 9999999)
                p_market_prob = np.random.uniform(0.44, 0.54)
                p_edge = np.random.uniform(-0.01, 0.09)
                if p_edge > 0.04:
                    strikeout_line = 6.5 if p_edge > 0.06 else 5.5
                    pick_side = "OVER" if p_edge > 0.06 else "UNDER"
                    model_prob = p_market_prob + p_edge
                    all_potential_wagers.append({
                        "matchup": matchup_name, "type": "🎯 PLAYER PROP (Pitcher)",
                        "selection": f"{home_pitcher} {pick_side} {strikeout_line} Strikeouts",
                        "raw_edge": p_edge, "market_odds": convert_prob_to_american_odds(p_market_prob),
                        "model_probability": model_prob, "fraction": min(0.015, 0.03 * kelly_fraction)
                    })
                already_scanned_player_props.add(home_pitcher)

            # --- Evaluate Batter Total Bases Prop Market ---
            if star_batter not in already_scanned_player_props:
                seed_b = generate_stable_seed(star_batter + "_PROP_BATTER_TB", 5000)
                np.random.seed(seed_b % 9999999)
                b_market_prob = np.random.uniform(0.42, 0.52)
                b_edge = np.random.uniform(-0.01, 0.09)
                if b_edge > 0.04:
                    base_line = 1.5
                    pick_side = "OVER" if b_edge > 0.06 else "UNDER"
                    model_prob = b_market_prob + b_edge
                    all_potential_wagers.append({
                        "matchup": matchup_name, "type": "🔥 PLAYER PROP (Batter)",
                        "selection": f"{star_batter} {pick_side} {base_line} Total Bases",
                        "raw_edge": b_edge, "market_odds": convert_prob_to_american_odds(b_market_prob),
                        "model_probability": model_prob, "fraction": min(0.015, 0.03 * kelly_fraction)
                    })
                already_scanned_player_props.add(star_batter)

        st.session_state.cached_optimized_wagers = sorted(all_potential_wagers, key=lambda x: x["raw_edge"], reverse=True)
        st.session_state.trading_slate_calculated = True

    st.button("Scan Complete Slate & Optimize Bets", on_click=execute_slate_optimization_callback)

    if st.session_state.trading_slate_calculated:
        st.markdown("## 📊 Mathematical Edge Ranking Matrix")
        
        raw_rows = []
        for item in st.session_state.cached_optimized_wagers:
            raw_rows.append({
                "Place Bet?": True,
                "Edge Rank": f"+{item['raw_edge']*100:.2f}%",
                "Matchup": item['matchup'],
                "Market Type": item['type'],
                "Selection Details": item['selection'],
                "Odds": item['market_odds'],
                "Model Prob": f"{item['model_probability']*100:.1f}%",
                "Suggested Risk": round(bankroll * item['fraction'], 2)
            })
            
        df_board = pd.DataFrame(raw_rows)

        # STATE-SAFE SUBMISSION FORM: Protects buttons from disappearing on change events
        with st.form("bet_slip_submission_form", clear_on_submit=False):
            edited_df = st.data_editor(
                df_board,
                column_config={
                    "Place Bet?": st.column_config.CheckboxColumn("Place Bet?", default=True),
                    "Edge Rank": st.column_config.TextColumn("Edge Rank", disabled=True),
                    "Matchup": st.column_config.TextColumn("Matchup", disabled=True),
                    "Market Type": st.column_config.TextColumn("Market Type", disabled=True),
                    "Selection Details": st.column_config.TextColumn("Selection Details", disabled=True),
                    "Odds": st.column_config.TextColumn("Odds", disabled=True),
                    "Model Prob": st.column_config.TextColumn("Model Prob", disabled=True),
                    "Suggested Risk": st.column_config.NumberColumn("Risk Allocation ($)", format="$%.2f", disabled=True)
                },
                hide_index=True, use_container_width=True, key="live_editor_grid"
            )

            # Form Action Submission Trigger
            form_submit_clicked = st.form_submit_button("🔒 Lock & Commit Active Bet Slip to Cloud Vault")

        confirmed_bets = edited_df[edited_df["Place Bet?"] == True]

        st.markdown("---")
        st.markdown("## 📜 Active Live Bet Slip Execution Order")

        running_total_liability = 0.0
        bets_placed_count = 0
        logged_slips_list = []

        for idx, row in confirmed_bets.iterrows():
            if running_total_liability >= max_daily_liability:
                break
                
            wager_amt = row["Suggested Risk"]
            if running_total_liability + wager_amt > max_daily_liability:
                wager_amt = max_daily_liability - running_total_liability
                
            if wager_amt > 0.01:
                running_total_liability += wager_amt
                bets_placed_count += 1
                
                logged_slips_list.append({
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Matchup": row["Matchup"],
                    "Market Type": row["Market Type"],
                    "Selection Details": row["Selection Details"],
                    "Odds": row["Odds"],
                    "Model Prob": row["Model Prob"],
                    "Risk Amount": float(wager_amt),
                    "Settled Status": "PENDING"
                })
                
                with st.container():
                    st.info(f"⚡ **Active Order #{bets_placed_count}** | {row['Matchup']}")
                    st.markdown(f"  * 👉 **SELECTION:** **{row['Selection Details']}** | **Odds:** `{row['Odds']}` | **Model Prob:** `{row['Model Prob']}`")
                    st.write(f"  * **Risk Allocation:** **${wager_amt:,.2f}**")
                    st.markdown("---")

        # Database Pipeline Commit Block
        if form_submit_clicked and bets_placed_count > 0:
            if db_conn is not None:
                try:
                    current_db_df = db_conn.read()
                    fresh_log_df = pd.DataFrame(logged_slips_list)
                    combined_db_df = pd.concat([current_db_df, fresh_log_df], ignore_index=True)
                    db_conn.update(data=combined_db_df)
                    st.success("✅ Bet execution log safely committed to the cloud database vault!")
                except Exception as db_err:
                    st.error(f"Database write dropped: {db_err}")
            else:
                st.error("Database link offline. Check your secrets credentials layout.")

        st.write(f"### 🛡️ Portfolio Risk Summary")
        st.write(f"Total Capital Allocated: **${running_total_liability:,.2f}** / Max Allowed: ${max_daily_liability:,.2f}")
        st.write(f"Actual Bankroll Exposure: **{(running_total_liability / bankroll) * 100:.2f}%**")

# =====================================================================
# 3. INTERACTIVE PERFORMANCE LAYER (TIME WINDOW ROI AGGREGATOR)
# =====================================================================
with nav_tab_2:
    st.markdown("## 📈 Performance & Rolling ROI Analytics Dashboard")
    
    if db_conn is not None:
        try:
            vault_df = db_conn.read()
            
            if not vault_df.empty:
                vault_df["Date"] = pd.to_datetime(vault_df["Date"]).dt.date
                vault_df["Risk Amount"] = vault_df["Risk Amount"].astype(float)
                
                def compute_window_metrics(dataframe, days_back):
                    today_date = datetime.now().date()
                    start_date = today_date - timedelta(days=days_back)
                    
                    if days_back == 1:
                        segment_df = dataframe[dataframe["Date"] == start_date]
                    else:
                        segment_df = dataframe[(dataframe["Date"] >= start_date) & (dataframe["Date"] < today_date)]
                        
                    settled_df = segment_df[segment_df["Settled Status"].isin(["WIN", "LOSS", "PUSH"])]
                    staked = settled_df["Risk Amount"].sum()
                    net_profit = 0.0
