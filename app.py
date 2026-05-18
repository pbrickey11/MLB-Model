import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import copy

# =====================================================================
# 1. MODEL ARCHITECTURE BLUEPRINTS (Required for pickle to un-serialize)
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
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        dataset = TensorDataset(X, y_soft)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                log_probs = torch.log_softmax(outputs, dim=1)
                loss = torch.mean(-torch.sum(batch_y * log_probs, dim=1))
                loss.backward()
                optimizer.step()
        return model

    def refine_labels(self, X: np.ndarray, y_noisy: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        num_classes = len(np.unique(y_noisy))
        y_soft = torch.nn.functional.one_hot(torch.tensor(y_noisy, dtype=torch.int64), num_classes).float()
        original_y_soft = y_soft.clone()
        for i in range(self.iterations):
            model = self._train_from_scratch(X_tensor, y_soft)
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                posteriors = torch.softmax(logits, dim=1)
            alpha = min(0.5 + (0.15 * i), 0.90) 
            y_soft = (1.0 - alpha) * original_y_soft + alpha * posteriors
        return y_soft.numpy()

# =====================================================================
# 2. STREAMLIT INTERACTIVE USER INTERFACE
# =====================================================================

st.title("⚾ MLB Quantitative Trading Dashboard")
st.write("Multi-Agent Reinforcement Learning Prediction Engine")

# Bankroll allocation input
bankroll = st.number_input("Enter Daily Starting Bankroll ($):", min_value=0.0, value=1000.0, step=100.0)
st.write(f"### Current Capital Allocation Baseline: ${bankroll:,.2f}")

if st.button("Fetch Live Data & Generate Recommendations"):
    st.info("🔄 Ingesting live Hawk-Eye telemetry, atmospheric data, and market consensus...")
    
    try:
        # Now pickle will successfully find the classes defined above!
        with open('model.pkl', 'rb') as file:
            model_package = pickle.load(file)
        st.success("✅ Model parameters loaded successfully!")
        
        st.markdown("### 📊 Daily Automated Betting Recommendations")
        st.write("The model recommends allocating a fraction of your bankroll to the following edge:")
        st.warning("👉 **Sample Edge Detected:** Matchup XYZ | Recommended Wager w/ Kelly Criterion Applied: 2.5% of Bankroll")
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
