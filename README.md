# Taxi Q-Learning Agent

This repository implements a tabular Q-learning agent for the `Taxi-v3` environment from OpenAI Gymnasium. It includes baseline training, hyperparameter tuning, performance analysis, and Q-value visualization.

## Directory Structure

```
.
├── baseline/                         # Results from default training configuration
│   ├── q_table.csv                   # Final Q-table from baseline run
│   └── training_log.csv              # Episode rewards during baseline training
├── hyperparameter_tuning/            # Grid search experiments
│   ├── grid_search_results.csv       # Summary of tuning runs
│   ├── q_tables/                     # Q-tables from various tuning configs
│   └── training_logs/                # Episode logs for each config
├── analysis.ipynb                    # Exploratory analysis of training performance
├── clean_up.sh                       # Utility script for clearing generated data
├── main.py                           # Entrypoint: trains agent and launches visual demo
├── README.md                         # Project overview and instructions
├── requirements.txt                  # Python dependencies
├── taxi_q_learner_optimizer.py       # Grid search utilities for hyperparameter tuning
├── taxi_q_learner.py                 # Q-learning agent logic
└── taxi_q_visualizer.py              # Q-table visualization on 5×5 grid
```

## Project Overview

The `Taxi-v3` environment simulates a 5×5 grid world where a taxi must:

- Pick up a passenger from one of four predefined locations (R, G, Y, B)
- Drop them off at a target location
- Learn to navigate efficiently while minimizing penalties and steps

This project uses **tabular Q-learning** to train an agent to solve this task, and includes tools for:

- Training and evaluating the agent
- Tuning hyperparameters via grid search
- Visualizing learned Q-values
- Analyzing performance trends

Environment reference: https://gymnasium.farama.org/environments/toy_text/taxi/

## Components

### Q-Learning Agent (`taxi_q_learner.py`)

- Implements tabular Q-learning
- Supports adjustable hyperparameters:
  - Learning rate (`alpha`)
  - Discount factor (`gamma`)
  - Epsilon decay schedule (`epsilon`, `min_epsilon`, `decay_start`)
- Logs reward per episode and saves final Q-table

### Hyperparameter Tuning (`taxi_q_learner_optimizer.py`)

- Performs exhaustive grid search over multiple hyperparameter combinations
- Saves logs and Q-tables for each run
- Logs results in `hyperparameter_tuning/grid_search_results.csv`

### Q-Table Visualization (`taxi_q_visualizer.py`)

- Randomly selects a state and displays a 5×5 grid
- Each cell shows directional Q-values (N/S/E/W) using green-shaded triangles
- Highlights:
  - Taxi position (`T`)
  - Passenger location (`P`)
  - Destination (`D`)

### Analysis (`analysis.ipynb`)

- Plots training reward curves
- Compares baseline and tuned models
- Evaluates convergence and stability

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```


### Run Baseline Training and Demo

```bash
python main.py
```

This will:

* Train the agent using a default configuration
* Save `q_table.csv` and `training_log.csv` in `baseline/`
* Launch an interactive demo using the learned policy


### Analyze Training Results

Open `analysis.ipynb` in Jupyter:

```bash
jupyter notebook analysis.ipynb
```

## Output Summary

* `baseline/`: Default training results
* `hyperparameter_tuning/`: Data from grid search experiments
* `analysis.ipynb`: Plots and comparisons

