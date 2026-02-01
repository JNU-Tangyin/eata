# Experiment Style & Visual Mapping Strategy

## 1. Design Philosophy
- **Aesthetic**: Minimalist, "High-Fashion" Academic (Tufte-like simplicity).
- **Palette**: Cool tones for methods, Warm tones for highlights/risks. High contrast for main results.
- **Layout**: Clear visual hierarchy, no overlapping modules, generous whitespace.

## 2. Visual Element Mapping

| Context | Visual Element | Style Specification | Meaning/Rationale |
| :--- | :--- | :--- | :--- |
| **Methods** | **Line/Bar Color** | **EATA (Ours)**: `RoyalBlue` (#4169E1) <br> **Baselines**: `Gray` (#808080) <br> **S&P 500**: `Black` (#000000) | Highlight our method against a muted background of baselines. |
| **Performance** | **Line Style** | **EATA**: Solid, Thick (2.5pt) <br> **Baselines**: Dashed/Dotted, Thin (1.0pt) | Emphasize stability and dominance of EATA. |
| **Metrics** | **Color Semantic** | **Profit/Return**: `Emerald` (#50C878) <br> **Loss/Risk**: `Crimson` (#DC143C) | Universal financial semiotics (Green=Good, Red=Bad). |
| **Data Points** | **Shapes** | **EATA**: `Star` ($\star$) or `Solid Circle` ($\bullet$) <br> **Baselines**: `Triangle` ($\triangle$) or `Cross` ($\times$) | Stars denote "Best/Selected". |
| **Text** | **Font** | **Axis Labels**: `Helvetica/Arial` (Sans-serif) <br> **Numbers**: `Times New Roman` (Serif, matching text) | Clean legibility for labels, consistency for data. |
| **Formulas** | **Annotation** | **Size**: `\footnotesize` or `8pt` <br> **Background**: `Yellow!10` (Pale highlight) | Subtle highlighting for discovered math expressions. |
| **Regimes** | **Background Areas** | **Bull**: `Green!5` <br> **Bear**: `Red!5` <br> **Volatile**: `Gray!10` | Contextual shading to show market conditions without clutter. |

## 3. Figure-Specific Mappings

### Fig 1: Cumulative Returns (Main Result)
- **Type**: Time Series Line Chart
- **X-Axis**: Time (Jan 2020 - Dec 2024)
- **Y-Axis**: Cumulative Return (%)
- **Mapping**:
    - EATA: Thick RoyalBlue Line
    - Buy & Hold: Thin Black Line
    - LSTM: Thin Dashed Gray Line
    - **Goal**: Show EATA's consistent outperformance and drawdown resilience.

### Fig 2: Pareto Frontier (Interpretability vs Performance)
- **Type**: Scatter Plot
- **X-Axis**: Complexity (Expression Length / Node Count)
- **Y-Axis**: Sharpe Ratio
- **Mapping**:
    - Points: Each dot is a model/run.
    - EATA Frontier: Connected line of top-left points (Low Complexity, High Sharpe).
    - Color Gradient: Darker = More Profitable.
    - **Goal**: Show EATA finds *simpler* models that perform *better* (breaking the trade-off).

### Fig 3: Return Distribution (Wasserstein Validation)
- **Type**: Density Plot (KDE)
- **X-Axis**: Daily Return %
- **Y-Axis**: Density
- **Mapping**:
    - Actual Data: Shaded Gray Area
    - EATA Prediction: Blue Line (Matches tail shape)
    - MSE-based Prediction: Red Dashed Line (Too Gaussian/Narrow)
    - **Goal**: Visual proof of the "Why Wasserstein" argument.

### Fig 4: Search Efficiency
- **Type**: Line Chart with Confidence Bands
- **X-Axis**: Wall-clock Time (Minutes)
- **Y-Axis**: Best Found Reward (Sharpe)
- **Mapping**:
    - EATA: Steep ascent (Blue)
    - GP: Slow ascent (Green)
    - Random: Flat (Gray)
    - **Goal**: Prove MCTS + Neural Guidance is faster than evolutionary methods.

## 4. LaTeX Style Definitions
```latex
\definecolor{eataBlue}{HTML}{4169E1}
\definecolor{profitGreen}{HTML}{50C878}
\definecolor{riskRed}{HTML}{DC143C}
\newcommand{\best}[1]{\textbf{\textcolor{eataBlue}{#1}}} % Highlight best results
```
