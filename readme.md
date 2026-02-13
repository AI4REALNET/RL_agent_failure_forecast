This repository implements a **Machine Learning pipeline** to assess power grid security by predicting the probability of cascading failures.

It leverages **Uncertainty Quantification (UQ)** to enhance risk assessment, separating uncertainty into:
1.  **Aleatoric Uncertainty:** Derived from load/generation forecasting residuals (HistGradientBoosting).
2.  **Epistemic Uncertainty:** Derived from an Evidential Neural Network (ENN) observing the grid state.

Finally, a **Classifier** predicts if disconnecting a specific critical line will cause a blackout, based on the current grid statistics and the quantified uncertainty.

**Supported Environments:**
* IEEE 14-bus (`l2rpn_case14_sandbox`)
* Network 36 (`l2rpn_icaps_2021_small`)

---

##  Project Structure

```text
grid_security_project/
│
├── data/                   # Generated datasets (CSV, NPY)
├── models/                 # Saved models (PKL, PTH)
├── agent/                  # Pre-trained Grid2Op Agent
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Central Configuration (Env, Modes, Paths)
│   ├── models.py           # Evidential Network (ENN) Architecture
│   ├── utils.py            # Feature extraction & Grid statistics
│
├── train_forecaster.py   # Module: Forecaster Training
├── collect_data.py       # Module: Simulation & Data Collection
├── train_classifier.py   # Module: Classifier Training & Inference
│
├── run_pipeline.py         #  MAIN (Run this file)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Instalation 

# 1. Clone repository
```

```

# 2. Install dependencies
It is highly recommend to use a virtual environment of Python3.10.

```
pip install -r requirements.txt
```
# 3. Agent Setup
Ensure your pre-trained agent (e.g. CurriculumAgent) is located in `agent/` folder. You can configure the specific path in `src/config.py`


## Configuration
All settings are controlled via `src/config.py`. **You do not need to modify the logic scripts directly**.

# Select Environment
Switch between the 14-bus and 36-bus networks by changing the ACTIVE_ENV variable:

```python 
# src/config.py
ACTIVE_ENV = "l2rpn_case14_sandbox"  # or "lr2rpn_icaps_2021_small
```

## Usage & Execution Modes
Use the main script to run the pipeline. The behaviour depends on the flags set in ```src/config.py```.

# 1. Training Pipeline
Use this mode to train the Forecasters, collect simulation data, and train the final Classifier.
1. **Config**: Set `TRAIN_MODE = True` in `src/config.py`.
2. **Run**:
    ```
   python run_pipeline.py
   ```
3. **Outcome**: All models wil be trained and saved in the `models/` directory.


# 2. Testing & Inference
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

## Methodology

# Uncertainty Features 
- **Epistemic (Model Uncertainty):** Calculated using an Evidential Neural Network (ENN). High values indicate the grid is in a state the agent/model has rarely seen (Out-of-Distribution).
- **Aleatoric (Data Uncertainty):** Calculated using a **GAMLSS-like approach** (Regressor on squared residuals). High values indicate the load/generation is behaving erratically/unpredictably.

# Critical Lines
The system automatically monitors specific critical lines defined in `Config14` or `Config36`. 