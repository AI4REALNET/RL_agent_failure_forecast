# Forecast RL Agent Failure

This repository implements a framework to **quantify and predict the realiability of pre-trained Reinforcement Learning (RL) agents** used fot real-time congestion management in power grids. 

Assessing the reliability of AI-assisted decision support systems under unseen operating conditions is critical. This project anticipates unreliable AI recommendations and provides early warnings to human operators.

The pipeline integrates **Uncertainty Quantification (UQ)** to support risk-aware decision making by separating uncertainty into two components:
  - **Aleatoric Uncertainty:** The uncertainty of the forecasted values (predictive variance). It captures the inherent stochastic variability and forecast errors of load and generation, estimated by modeling the residuals of the primary Forecaster (HistGradientBoosting).
  - **Epistemic Uncertainty:** The uncertainty associated with the RL agent's decisions when facing out-of-distribution or unseen grid states, computed using an **Evidential Neural Network (ENN)**.



These indicators are integrated into a **failure prediction model** that estimates the probability of RL agent failure under future contigencies (disconnection of lines).

Finally, a **Dual LLM Architecture** takes the outputs of the predictive classifiers and synthesizes robust, symbolic Python rules (`best_rule.py`). This translates complex, black-box uncertainty metrics into intepretable, human-readable operational guidelines, ensuring the AI assistant's boundaries are transparent and safe.

---

## Supported Environments

- **Network 36** (`l2rpn_icaps_2021_small`)

---

## Project Structure

```text
grid_security_project/
│
├── agents/
│   └── network36/                       # Pre-trained Grid2Op agent (CurriculumAgent)
│
│
├── forecasts/
│   └── HBGB_36.pkl                      # Load / generation forecaster
│
├── models/
│   ├── enn_36.pth                       # Evidential Neural Network (epistemic uncertainty)
│   ├── HBGB_36_aleatoric.pkl            # Aleatoric uncertainty model
│
├── llm_rules_results/                   # Generated Symbolic Rules (Dual LLM output)
│   └── temp_<value>/                    # Explore different LLM temperatures (e.g., temp_0.3, temp_0.5)
│       └── line_<line_id>/              # Results for a specific critical line - line_id
│           ├── best_rule.py             # The final interpretable Python rule
│   
├── src/
│   ├── collect_data.py                  # Simulation and dataset generation
│   ├── config.py                        # Central configuration (environment, modes, paths)
│   ├── dual_llm.py                      # Dual LLM: generator and critic 
│   ├── enn_models.py                    # ENN architectures
│   ├── rule_predictor.py                # Live rule inference and natural-language translation
│   ├── train_classifier.py              # Classifier training and inference
│   ├── train_forecast.py                # Forecaster training
│   ├── training_enn.py                  # ENN training pipeline
│   └── utils.py                         # Feature extraction and grid statistics
│
├── run_pipeline.py                      # Main entry point
├── requirements.txt                     # Python dependencies
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
Ensure your pre-trained agent (e.g. CurriculumAgent) is located in `agent/` folder. For this, go to the https://github.com/AI4REALNET/RL_agent_failure_forecast/releases/tag/v1.0-agents of this repository and download. You can configure the specific path in `src/config.py`. 


### Configuration
All settings are controlled via `src/config.py`. **You do not need to modify the logic scripts directly**.

#### Select Environment
Switch between the 14-bus and 36-bus networks by changing the ACTIVE_ENV variable:

```python 
# src/config.py
ACTIVE_ENV = "lr2rpn_icaps_2021_small"
```

### 5. Pre-trained Models

The forecaster models are too large to be stored directly in this repository.
They are hosted as binary attachments in the GitHub Releases section.

1. Go to the https://github.com/AI4REALNET/RL_agent_failure_forecast/releases/tag/v1.0-models
   of this repository.
2. Download the following files from release `v1.0-models`:
   - `HBGB_36.pkl` -> place in `forecasts/`
   - `HBGB_36_aleatoric.pkl` -> place in `models/`
   - `enn_36.pth` -> place in `models/`

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

C. **LLM Rule Inference (Natural-Language Explanations)**
Use this mode to apply the symbolic rules generated by the Dual LLM to a live simulation episode. For each monitored line, at every analysis step the system runs the forecast pipeline internally, evaluates the corresponding rule, and prints a human-readable explanation of the prediction.

1. *Config*:
```python
TRAIN_MODE          = False
TEST_SINGLE_EPISODE = False
PREDICT_PROBA_MODE  = False
LLM_RULE_MODE       = True

# Path to the folder containing the generated rules
LLM_RULES_DIR     = "llm_rules_results/temp_0.5"

# Episode seed to simulate
LLM_RULES_EPISODE = 50
```
2. *Run*: `python run_pipeline.py`
3. *Outcome*: For each monitored line and analysis step, prints the binary prediction (`OK` or `FAILURE PREDICTED`) together with a plain-English explanation sentence. At startup, all available rule sentences are also printed for use as operational guidelines.

Example output:
```
  [41_48_131]
  Following a contingency on line 41_48_131, the RL agent is predicted to fail
  to provide a recommendation that solves a congestion problem if the maximum
  line loading (rho) at t is >= 0.82, or if the forecasted maximum line loading
  (rho) at t+12 is >= 0.66 while the epistemic uncertainty at t is >= 0.77 and
  the forecasted total active power load at t+12 is <= 643 MW.

  --- Step 40 ---
    Line 41_48_131    -> FAILURE PREDICTED
      Following a contingency on line 41_48_131, ...
    Line 34_35_110    -> OK
```

#### Critical Lines
The system automatically monitors specific critical lines defined in CFG. 



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
conditions are, independently of the agent's knowledge.

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

### 5. LLM-Guided Symbolic Rule Generation (Dual LLM)

To convert the black-box classifier outputs into interpretable operational guidelines, a **Dual LLM Architecture** (Generator-Evaluator) processes the data. The system automatically iterates over multiple critical lines and explores various LLM temperature settings (hyperparameter search) to find the optimal balance between logical strictness and creative problem-solving.

- **Dynamic Rule Synthesis:** The generator LLM writes explicit, symbolic Python rules (`best_rule.py`) for each targeted transmission line based on thresholds of grid statistics and uncertainty metrics.
- **Evaluation & Refinement:** An evaluator LLM critiques the generated rules against false-alarm and oversight metrics. Changes and logical justifications are systematically logged (`best_feedback.txt`, `best_justification.txt`).
- **Iterative Tracking:** The framework iteratively tests seeds and records performance metrics across different temperature folders, ensuring convergence on the safest and most accurate operational rule for every monitored line.

---

### 6. Rule Translation (Natural-Language Explanations)

To make the symbolic rules accessible to human operators, `src/rule_predictor.py` automatically translates each `best_rule.py` into a plain-English sentence that describes the conditions under which the RL agent is predicted to fail.

The translation is performed by parsing the Python rule as an Abstract Syntax Tree (AST) and mapping each condition to a human-readable description of the corresponding grid feature. Each distinct failure path within the rule becomes an "or if" clause, and multiple AND conditions within the same path are joined with "while ... and".

For example, the following rule:

```python
def rule(x):
    if x["max_line_rho"] >= 0.65:
        if x["epistemic_before"] >= 0.7891:
            if x["aleatoric_gen_p_mean"] <= 0.3434:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        if x["fcast_sum_load_q"] >= 155.5429:
            if x["aleatoric_gen_p_mean"] <= 0.3434:
                return 1
            else:
                return 0
        else:
            return 0
```

is automatically translated to:

> *Following a contingency on line 34_35_110, the RL agent is predicted to fail if the maximum line loading at t is >= 0.65 while the epistemic uncertainty is <= 0.79 and the mean aleatoric generation uncertainty is <= 0.34, or if the forecasted reactive load at t+12 is >= 155.54 MVAR and the mean aleatoric generation uncertainty is <= 0.34$. 

In LLM Rule Inference mode, the system also evaluates each rule against the current grid state in real time: it runs the forecast pipeline internally (computing t+12 features from the live observation), applies the rule, and reports the prediction alongside the explanation sentence.

---

### Final Objective

The ultimate goal of this framework is to provide **real-time confidence levels**
that allow:

- Validation of autonomous agent decisions.
- Prevention of unsafe operations in critical power grid environments through transparent, human-readable guidelines.
