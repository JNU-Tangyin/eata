# EATA Paper: Next Steps for Top-Tier Submission

To meet the rigorous standards of journals like TPAMI, ICML, or Quantitative Finance, the following experimental and theoretical gaps must be addressed.

## 1. Scale & Generalization (Critical)
- [ ] **S&P 100/500 Experiment**: Expand evaluation from 31 stocks to the full S&P 100 or S&P 500 constituents.
    - *Goal*: Prove robust generalization across sectors and market caps.
    - *Metric*: Report average Rank IC (Information Coefficient) across the universe.
- [ ] **Regime Analysis**: Stratify performance by market regimes (Bull, Bear, Sideways, High Volatility).
    - *Goal*: Demonstrate where EATA excels and where it fails (e.g., confirm failure in "meme" volatility).

## 2. Theoretical Validation (Methodology)
- [ ] **Wasserstein vs. MSE Visualization**:
    - Create a figure showing the density of Actual Returns vs. Predicted Returns (EATA) vs. Predicted Returns (MSE-based).
    - *Goal*: Visually prove that EATA captures the "shape" (tails) better than MSE.
- [ ] **Profit Head Correlation**:
    - Calculate the correlation between the Profit Head's output ($P_\theta$) and the actual realized RL reward ($r^{rl}$).
    - *Goal*: Prove the Profit Head is actually learning to predict trainability/profitability, not just noise.

## 3. Baseline Rigor (Experiments)
- [ ] **Deep Learning Tuning**:
    - Re-run LSTM, Transformer, and PPO baselines with the specific grid search mentioned in the paper:
        - LR: $\{10^{-3}, 10^{-4}, 10^{-5}\}$
        - Hidden Dim: $\{64, 128, 256\}$
    - *Goal*: Eliminate the suspicion that baselines were "weakened".
- [ ] **Significance Testing**:
    - Explicitly state the null hypothesis for t-tests (e.g., $H_0: \mu_{EATA} \leq \mu_{Baseline}$).
    - Add Bonferroni correction if multiple comparisons are heavily emphasized.

## 4. Sensitivity & Hyperparameters
- [ ] **Signal Threshold Analysis**:
    - Test sensitivity of the Trading Signal rule ($Q_{25}/Q_{75}$) vs. Median ($Q_{50}$) vs. Aggressive ($Q_{40}/Q_{60}$).
    - *Goal*: Justify the conservative $25/75$ choice.
- [ ] **Lookback Window**:
    - Test if performance degrades significantly with very long windows (e.g., 200 days) due to non-stationarity.

## 5. Ablation Studies
- [ ] **Grammar Augmentation**:
    - Analyze the *quality* of extracted modules. Do they correspond to known financial concepts (e.g., Bollinger Bands, MACD variations)?
    - *Goal*: Enhance interpretability argument.

## 6. Paper Polishing
- [ ] **Visual Consistency**: Ensure all plots use the exact same font style and size as the LaTeX text.
- [ ] **Notation Check**: Final sweep to ensure $T^{in}$ vs $T_{in}$ consistency (Standardize on $T^{\text{in}}$).
