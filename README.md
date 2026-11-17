# Atari Breakout with Proximal Policy Optimization (PPO)
**A Reinforcement Learning implementation of Atari Breakout using Proximal Policy Optimization (PPO), PyTorch and a custom Failure Memory mechanism.**

---

## 1. Project Overview

This project implements an intelligent agent capable of playing **Atari Breakout** using the **Proximal Policy Optimization (PPO)** algorithm, a policy-gradient reinforcement learning method.

The objective is to train an agent to interact with the Breakout environment, learn effective strategies through trial and error, and progressively improve its performance.

The implementation is developed from scratch using Python and PyTorch.

### Key Features

- Custom Atari Breakout environment with image preprocessing.
- Actor-Critic neural network architecture.
- PPO algorithm implementation.
- Experience buffer for storing training data.
- **Failure Memory:** an experimental mechanism designed to incorporate information from previous failures.
- Training performance monitoring.
- Model checkpoint saving and visual demonstration.

---

## 2. Technologies and Environment

| Component | Technology |
|---|---|
| Programming language | Python 3.10 |
| Deep Learning | PyTorch |
| Reinforcement Learning | PPO |
| Game environment | Gymnasium |
| Atari emulator | ALE (Arcade Learning Environment) |
| Numerical computation | NumPy |
| Image processing | OpenCV |
| Visualization | Matplotlib |
| Development environment | Visual Studio Code |

**Tested platform:** Windows 10 / Windows 11.

---

## 3. Project Architecture

The project is organized into independent modules to separate environment management, neural network architecture, reinforcement learning logic and experimentation.

```text
BREAKOUT-PPO/
│
├── config.py
├── env.py
├── model.py
├── ppo.py
├── memory.py
│
├── main.py
├── demo.py
├── test.py
│
├── ppo_breakout.pt
│
├── README.md
└── requirements.txt
```

### Module Description

| File | Responsibility |
|---|---|
| `config.py` | Defines training hyperparameters and configuration settings. |
| `env.py` | Creates the Breakout environment and handles image preprocessing. |
| `model.py` | Implements the Actor-Critic neural network architecture. |
| `ppo.py` | Implements the Proximal Policy Optimization algorithm. |
| `memory.py` | Handles the PPO experience buffer and the experimental Failure Memory mechanism. |
| `main.py` | Entry point for agent training. |
| `demo.py` | Loads a trained model and runs a visual demonstration. |
| `test.py` | Contains project testing routines. |

---

## 4. Reinforcement Learning Approach

### Proximal Policy Optimization (PPO)

PPO is an on-policy reinforcement learning algorithm that improves a policy while limiting excessively large policy updates.

The agent learns through interactions with the Atari Breakout environment.

The implementation uses an Actor-Critic architecture:

- **Actor:** determines the action probabilities.
- **Critic:** estimates the value of the current state.

The agent receives visual observations from the environment and learns to select actions that maximize cumulative rewards.

### Failure Memory — Experimental Contribution

This project also explores an additional mechanism called **Failure Memory**.

The objective is to incorporate information from previous unsuccessful experiences into the learning process.

This experimental component is intended to complement the standard PPO training process and investigate whether retaining information about failures can help improve the agent's learning behavior.

---

## 5. Installation and Setup

### Requirements

- Python 3.10
- pip
- Windows 10 or Windows 11

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/BREAKOUT-PPO.git
cd BREAKOUT-PPO
```

### Step 2 — Create a virtual environment

```powershell
py -3.10 -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\activate
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip

pip install torch torchvision
pip install "gymnasium[atari]"
pip install ale-py
pip install autorom
pip install numpy
pip install opencv-python
pip install matplotlib
```

### Step 4 — Install Atari ROMs

Run:

```bash
AutoROM --accept-license
```

This prepares the Atari ROMs required by the environment.

---

## 6. Training the Agent

To start training, execute:

```bash
python main.py
```

The training process:

1. Initializes the Breakout environment.
2. Creates the Actor-Critic model.
3. Initializes the PPO training components.
4. Collects experience through environment interactions.
5. Updates the policy using PPO.
6. Displays training performance.
7. Saves the trained model.

The trained model is saved as:

```text
ppo_breakout.pt
```

### Training Monitoring

The console displays training information, including the mean reward over recent episodes.

Example:

```text
Update X/Y | Mean Reward (last 10 ep): ...
```

---

## 7. Running the Agent Demonstration

Once training is complete, run:

```bash
python demo.py
```

The demonstration:

1. Loads the trained model.
2. Initializes the Breakout environment.
3. Runs the agent using its learned policy.
4. Displays the game visually.
5. Reports the reward obtained during each episode.

**Important:** The trained model `ppo_breakout.pt` must be available in the project root directory. The demonstration uses visual rendering (`render=True`).

---

## 8. Quick Start

For convenience, the main commands are:

| Action | Command |
|---|---|
| Create environment | `py -3.10 -m venv venv` |
| Activate environment | `venv\Scripts\activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Train the agent | `python main.py` |
| Run demonstration | `python demo.py` |

---

## 9. Future Improvements

Potential directions for further experimentation include:

- Evaluating the impact of Failure Memory on PPO performance.
- Comparing PPO with other reinforcement learning algorithms.
- Studying training stability and sample efficiency.
- Experimenting with different neural network architectures.
- Comparing reward progression across multiple training configurations.

---

## 10. Author

**Mame Tabara SY**

Master's Student in Computer Science  
Artificial Intelligence and Data Science  
Université de Sherbrooke  
Sherbrooke, Quebec, Canada

---

**Project:** Atari Breakout with PPO  
**Domain:** Artificial Intelligence / Reinforcement Learning  
**Language:** Python
