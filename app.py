import streamlit as st
import numpy as np
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import hashlib, urllib.request, json

# =====================================================================
# 1. CORE ARCHITECTURE BLUEPRINTS (Required for model initialization)
# =====================================================================
class SabermetricEngine:
    def __init__(self, xwoba_threshold=0.350, exit_velo_threshold=95.0):
        self.xwoba_threshold, self.exit_velo_threshold = xwoba_threshold, exit_velo_threshold
    def compute_swdecision_plus(self, xwoba, exit_velo, out_of_zone):
        res = np.where(xwoba >= self.xwoba_threshold, 1.25, -0.50)
        return res + np.where((out_of_zone == 1) & (exit_velo < self.exit_velo_threshold), -1.5, np.where((out_of_zone == 1) & (exit_velo >= self.exit_velo_threshold), 1.0, 0.0))

class BaseNeuralNetwork(nn.Module):
    def __init__(self, input_dim=5, output_classes=2):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, output_classes))
    def forward(self, x): return self.network(x)

class IAR2_Refiner:
    def __init__(self, base_classifier, iterations=3, lr=1e-3, batch_size=64):
        self.base_classifier, self.iterations, self.lr, self.batch_size = base_classifier, iterations, lr, batch_size
    def refine_labels(self, X, y_noisy: np.ndarray) -> np.ndarray: return np.array([[0.5, 0.5]])

def generate_stable_seed(string_input: str, offset: int) -> int:
    return int(hashlib.sha256(string_input.encode('utf-8')).hexdigest()[:8], 16) + offset

def convert_prob_to_american_odds(prob: float) -> str:
    if prob <= 0 or prob >= 1: return "+100"
    return f"-{int(round((prob / (1.0 - prob)) * 100.0))}" if prob > 0.50_f else f"+{int(round(((1.0 - prob) / prob) * 100.0))}"

def calculate_payout(odds_str: str, risk: float) -> float:
    try:
        odds = int(odds_str.replace("+", ""))
        return risk * (odds / 100.0) if odds > 0 else risk / (abs(odds) / 100.0)
    except Exception: return risk

# =====================================================================
# 2. RUNTIME GRAPHICAL INTERFACE LAYER
# =====================================================================
st.set_page_config(layout="wide")
st.title("⚾ MLB Quantitative Trading Dashboard")

# CRYPTOGRAPHIC VAULT RECONSTRUCTION: Manually builds a clean RSA key block in volatile memory
try:
    sec = st.secrets["connections"]["gsheets"]
    raw = sec["raw_key"].replace(" ", "").replace("\n", "")
    # Slice the solid string block into clean 64-character lines
    chunks = [raw[i:i+64] for i in range(0, len(raw), 64)]
    formatted_key = "-----BEGIN PRIVATE KEY-----\n" + "\n".join(chunks) + "\n-----END PRIVATE KEY-----\n"
    
    # Overwrite the credential map in memory before initializing the gsheets driver
    st.secrets["connections"]["gsheets"]["private_key"] = formatted_key
except Exception as parse_error:
    st.error(f"Cryptographic reconstruction failed: {parse_error}")

try:
    db_conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"Google Sheets Connection Driver Error: {e}")
    db_conn = None

nav_tab_1, nav_tab_2 = st.tabs(["🚀 Live Edge Calculator", "📊 Financial Performance Audit Vault"])

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
    "St. Louis Cardinals": {"pitcher": "Sonny Gray", "batter": "Nolan Arenado"}
}

with nav_tab_1:
    bankroll = st.number_input("Enter Daily Starting Bankroll ($):", min_value=0.0, value=1000.0, step=100.0)
    kelly_fraction = st.sidebar.slider("Kelly Criterion Modifier", 0.10, 1.00, 0.25, step=0.05)
    daily_max_exposure_pct = st.sidebar.slider("Max Total Daily Bankroll Exposure (%)", 5.0, 25.0, 10.0, step=1.0)
    
    max_daily_liability = bankroll * (daily_max_exposure_pct / 100.0)
    st.write(f"### Current Capital Baseline: ${bankroll:,.2f} | Max Liability Limit: ${max_daily_liability:,.2f}")

    if "cached_optimized_wagers" not in st.session_state:
        st.session_state.cached_optimized_wagers = []

    if st.button("🚀 Run Analytical Model Scanner", type="primary"):
        mock_slate = [("New York Yankees", "Boston Red Sox"), ("Los Angeles Dodgers", "San Francisco Giants"), ("Cincinnati Reds", "St. Louis Cardinals")]
        wagers = []
        for home, away in mock_slate:
            seed = generate_stable_seed(home + "_ML", 1000)
            np.random.seed(seed % 9999999)
            prob, edge = np.random.uniform(0.45, 0.55), np.random.uniform(0.04, 0.08)
            wagers.append({
                "matchup": f"{away} @ {home}", "type": "🔹 MONEYLINE", "selection": home,
                "raw_edge": edge, "market_odds": convert_prob_to_american_odds(prob),
                "model_probability": prob + edge, "fraction": min(0.04, 0.05 * kelly_fraction)
            })
        st.session_state.cached_optimized_wagers = sorted(wagers, key=lambda x: x["raw_edge"], reverse=True)

    if st.session_state.cached_optimized_wagers:
        raw_rows = [{
            "Place Bet?": True, "Edge Rank": f"+{item['raw_edge']*100:.2f}%", "Matchup": item['matchup'],
            "Market Type": item['type'], "Selection Details": item['selection'], "Odds": item['market_odds'],
            "Model Prob": f"{item['model_probability']*100:.1f}%", "Suggested Risk": round(bankroll * item['fraction'], 2)
        } for item in st.session_state.cached_optimized_wagers]
        
        edited_df = st.data_editor(pd.DataFrame(raw_rows), hide_index=True, use_container_width=True, key="live_editor_grid")
        confirmed = edited_df[edited_df["Place Bet?"] == True]

        st.markdown("---")
        st.markdown("## 📜 Active Live Bet Slip Execution Order")
        
        running_liab, slips = 0.0, []
        for _, row in confirmed.iterrows():
            wager = row["Suggested Risk"]
            if running_liab + wager > max_daily_liability: wager = max_daily_liability - running_liab
            if wager > 0.01:
                running_liab += wager
                slips.append({"Date": datetime.now().strftime("%Y-%m-%d"), "Matchup": row["Matchup"], "Market Type": row["Market Type"], "Selection Details": row["Selection Details"], "Odds": row["Odds"], "Model Prob": row["Model Prob"], "Risk Amount": float(wager), "Settled Status": "PENDING"})
                st.info(f"⚡ **Active Order** | {row['Matchup']} | Selection: {row['Selection Details']} | Risk: ${wager:,.2f}")

        if slips and st.button("🔒 Lock & Commit Active Bet Slip to Cloud Vault"):
            if db_conn is not None:
                try:
                    curr = db_conn.read()
                    db_conn.update(data=pd.concat([curr, pd.DataFrame(slips)], ignore_index=True))
                    st.success("✅ Bet execution log safely committed to cloud!")
                except Exception as e: st.error(f"Database write dropped: {e}")
            else: st.error("Database connection configuration missing.")

# =====================================================================
# 3. FINANCIAL ACCOUNTING AND AUDIT LEDGER LAYER
# =====================================================================
with nav_tab_2:
    st.markdown("## 📈 Performance & Rolling ROI Analytics Dashboard")
    if db_conn is not None:
        try:
            vault_df = db_conn.read()
            if not vault_df.empty:
                vault_df["Date"] = pd.to_datetime(vault_df["Date"]).dt.date
                vault_df["Risk Amount"] = vault_df["Risk Amount"].astype(float)
                
                def get_metrics(df, days):
                    start = datetime.now().date() - timedelta(days=days)
                    sub = df[df["Date"] == start] if days == 1 else df[(df["Date"] >= start) & (df["Date"] < datetime.now().date())]
                    settled = sub[sub["Settled Status"].isin(["WIN", "LOSS", "PUSH"])]
                    staked = settled["Risk Amount"].sum()
                    prof = sum([-r["Risk Amount"] if r["Settled Status"] == "LOSS" else (calculate_payout(str(r["Odds"]), r["Risk Amount"]) if r["Settled Status"] == "WIN" else 0.0) for _, r in settled.iterrows()])
                    return staked, prof, (prof / staked * 100.0 if staked > 0 else 0.0)

                m1, m2, m3 = st.columns(3)
                for m, d, name in [(m1, 1, "Yesterday's"), (m2, 7, "Prior 7-Day"), (m3, 30, "Prior 30-Day")]:
                    stk, prf, roi = get_metrics(vault_df, d)
                    with m:
                        st.markdown(f"### {name} Summary")
                        st.metric("Total Staked", f"${stk:,.2f}")
                        st.metric("Net Returns", f"${prf:,.2f}", f"{roi:.1f}% ROI")

                st.markdown("---")
                up_df = st.data_editor(vault_df, hide_index=True, use_container_width=True, key="vault_audit_matrix", column_config={"Settled Status": st.column_config.SelectboxColumn("Settled Status", options=["PENDING", "WIN", "LOSS", "PUSH"], required=True)})
                if st.button("💾 Save Settled Ledger Changes"):
                    db_conn.update(data=up_df)
                    st.success("✅ Audit values committed!")
                    st.rerun()
            else: st.info("Tracking vault ledger is currently empty.")
        except Exception as e: st.error(f"Error accessing vault entries: {e}")
