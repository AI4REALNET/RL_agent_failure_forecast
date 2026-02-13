# Forecast RL Agent Failure

This repository implements a **Machine Learning pipeline** for **power grid security assessment**.  
The objective is to predict the probability of cascading failures caused by the disconnection of critical transmission lines.

The pipeline integrates **Uncertainty Quantification (UQ)** to support risk-aware decision making by separating uncertainty into:

- **Aleatoric Uncertainty**  
  Inherent variability in load and generation forecasts, estimated using residual-based models (HistGradientBoosting).

- **Epistemic Uncertainty**  
  Model uncertainty due to lack of knowledge, computed using an **Evidential Neural Network (ENN)** observing the grid state.

A **final binary classifier** combines grid statistics with uncertainty features to predict whether disconnecting a given line will cause a blackout.

---

## Supported Environments

- **IEEE 14-bus** (`l2rpn_case14_sandbox`)
- **Network 36** (`l2rpn_icaps_2021_small`)

---

## Project Structure

```text
grid_security_project/
│
├── agents/
│   └── network36/                # Pre-trained Grid2Op agent (CurriculumAgent)
│
├── data/
│   ├── uncertainty_disconnection_analysis.csv
│   ├── X_train_36.npy
│   └── y_train_36.npy
│
├── forecasts/
│   └── HBGB_36.pkl                # Load / generation forecaster
│
├── models/
│   ├── enn_36.pth                 # Evidential Neural Network (epistemic uncertainty)
│   ├── enn_data/                  # ENN training datasets
│   ├── HBGB_36_aleatoric.pkl       # Aleatoric uncertainty model
│   └── final_classifier36.pkl     # Final blackout classifier
│
├── src/
│   ├── collect_data.py            # Simulation and dataset generation
│   ├── config.py                  # Central configuration (environment, modes, paths)
│   ├── enn_models.py              # ENN architectures
│   ├── train_classifier.py        # Classifier training and inference
│   ├── train_forecast.py          # Forecaster training
│   ├── training_enn.py            # ENN training pipeline
│   └── utils.py                   # Feature extraction and grid statistics
│
├── run_pipeline.py                # Main entry point
├── requirements.txt               # Python dependencies
```
## Instalation 

### 1. Python version
This project requires **Python 3.10.13**.

Verify your Python version:
```
python --version
# Python 3.10.13

```
### 2. Clone the Repository
```
git clone <repository_url>
```

### 3. Install dependencies
It is highly recommend to use a virtual environment of Python3.10.

```
pip install -r requirements.txt
```
### 4. Agent Setup
Ensure your pre-trained agent (e.g. CurriculumAgent) is located in `agent/` folder. You can configure the specific path in `src/config.py`


### Configuration
All settings are controlled via `src/config.py`. **You do not need to modify the logic scripts directly**.

#### Select Environment
Switch between the 14-bus and 36-bus networks by changing the ACTIVE_ENV variable:

```python 
# src/config.py
ACTIVE_ENV = "l2rpn_case14_sandbox"  # or "lr2rpn_icaps_2021_small
```

#### Usage & Execution Modes
Use the main script to run the pipeline. The behaviour depends on the flags set in ```src/config.py```.

##### 1. Training Pipeline
Use this mode to train the Forecasters, collect simulation data, and train the final Classifier.
1. **Config**: Set `TRAIN_MODE = True` in `src/config.py`.
2. **Run**:
    ```
   python run_pipeline.py
   ```
3. **Outcome**: All models wil be trained and saved in the `models/` directory.


##### 2. Testing & Inference
If you have trained models, you can use the following modes to test the system.

A. **Single Episode Simulation**
Use this to analyse a specific episode (seed) from start to finish. It simulates the agent interacting with the grid and records how the uncertainty metrics behave over time.
1. *Config*:
```python
TRAIN_MODE = False
TEST_SINGLE_EPISODE = True
EPISODE_ID_TO_TEST = 50 # The seed of the episode
```
2. *Run*: python run_pipeline.py
3. *Outcome*: Generates a CSV trace of that specific episode in `data/`.

B. **Single Observation Inference (Probabilities)**
Use this to predict the failure probability for **one specific grid state** (Observation). This mode **does not** run a physical simulation (no disconnection). It purely calculates risk based on the model's knowledge.
1. *Config*:
```python
TRAIN_MODE = False
TEST_SINGLE_EPISODE = FALSE
PREDICT_PROBA_MODE = True

# Define which state to fetch from the environment
PROBA_TEST_EPISODE_ID = 50
PROBA_TEST_STEP = 50
```
2. *Run*: python run_pipeline.py
3. *Outcome*: Generates a CSV trace of that specific episode in `data/`.

#### Critical Lines
The system automatically monitors specific critical lines defined in `Config14` or `Config36`. 

## Methodology

This work proposes a **failure probability forecasting framework** that combines
**power grid forecasting**, **uncertainty quantification**, and **risk classification**
to anticipate cascading failures caused by line disconnections.

The methodology is structured into four main stages.

---

### 1. Uncertainty Decomposition

The framework explicitly separates uncertainty into **aleatoric** and **epistemic**
components, each capturing different sources of risk.

#### Aleatoric Uncertainty (Data Uncertainty)

Aleatoric uncertainty captures the **stochastic variability** inherent to load and
generation dynamics.

- A **multi-output time-series forecaster** (HistGradientBoosting) predicts active and
  reactive power injections for all loads and generators.
- Forecasts are generated for a **1-hour horizon** (12 timesteps ahead).
- Squared residuals between ground-truth values and mean forecasts are computed.
- A secondary regression model is trained on these squared residuals to estimate
  the **forecast variance**, which is used as a proxy for aleatoric uncertainty.

This process allows the framework to quantify how unpredictable future operating
conditions are, independently of the agent’s knowledge.

---

#### Epistemic Uncertainty (Model Uncertainty)

Epistemic uncertainty reflects the **lack of knowledge of the agent** about the current
grid state and is used as an indicator of **out-of-distribution (OOD)** situations.

- An **Evidential Neural Network (ENN)** is trained via **knowledge distillation** to
  replicate the policy of a Senior (expert) agent.
- Instead of producing softmax probabilities, the ENN outputs the parameters
  of a **Dirichlet distribution** over the action space.
- Model ignorance is computed analytically as:

u = K / sum(alpha_i)

where `K` is the number of actions and `alpha_i` are the Dirichlet parameters.

High epistemic uncertainty indicates that the agent is operating in rarely observed or
unknown grid conditions.

---

### 2. Forecasting Future Grid States

To anticipate failures before they occur, the framework predicts **future grid states**.

- Load and generation forecasts are injected into the power grid model.
- A power flow simulation is executed to obtain the **forecasted grid state**
  one hour ahead.
- These future states are combined with aleatoric uncertainty estimates, capturing
  intrinsic forecast variability.

---

### 3. Contingency Analysis

For each candidate critical line:

- A **what-if disconnection** is simulated on the forecasted grid state.
- The system evaluates whether the grid remains stable or reaches a failure condition
  one hour after the contingency.
- This process generates labeled data linking grid conditions, uncertainties, and
  line disconnections to observed failures.

---

### 4. Risk Classification

A final **binary classifier** is trained to predict cascading failures **before action
execution**.

**Inputs:**
- Current grid state indicators (e.g., load-generation balance, thermal stress).
- Epistemic uncertainty (confidence in the current state).
- Aleatoric uncertainty (forecast variability).
- Identifier of the disconnected transmission line.

**Output:**
- `0` – Stable operation expected.
- `1` – Failure predicted (alarm triggered).

---

### Final Objective

The ultimate goal of this framework is to provide **real-time confidence levels**
that allow:

- Anticipation of cascading failures.
- Validation of autonomous agent decisions.
- Prevention of unsafe operations in critical power grid environments.

