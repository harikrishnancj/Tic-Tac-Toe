# Adaptive Tic-Tac-Toe Solver with Q-Learning + Heuristic Rules

## Overview

This project implements an **adaptive Tic-Tac-Toe agent** using **Q-Learning**, enhanced with **heuristic rules**. The agent learns optimal moves over time through self-play, while heuristics ensure it can immediately block or win when possible.  

Key features:
- Q-Learning based reinforcement learning agent
- Heuristic rules for immediate win or blocking opponent
- Interactive human vs AI gameplay
- Fully modular design (separate files for environment, agent, training, and gameplay)

---

## Table of Contents

1. [Environment](#environment)
2. [QLearning Agent](#qlearning-agent)
3. [Training Logic](#training-logic)
4. [Heuristic Rules](#heuristic-rules)
5. [Gameplay](#gameplay)
6. [Q-Learning Equation](#q-learning-equation)
7. [How to Run](#how-to-run)

---

## Environment

The Tic-Tac-Toe environment (`env.py`) represents the 3x3 board and handles:

- **Board State**: Flattened into a tuple for easy use as a Q-table key.
- **Available Moves**: Returns indices of empty cells.
- **Step Function**: Executes a move, checks for win/draw, returns:
  - `next_state`
  - `reward`  
  - `done` (if the game is over)
- **Render**: Pretty-prints the board in console.

### Board Mapping (Flattened):


- X → 1  
- O → -1  
- Empty → 0  

---

## QLearning Agent

The agent (`agent.py`) uses **Q-Learning**:

- **Q-table**: Maps `(state)` → `[Q-value for each of 9 positions]`
- **Learning Rate (α)**: How fast it updates Q-values
- **Discount Factor (γ)**: Importance of future rewards
- **Epsilon (ε)**: Probability to explore random moves  
- **Epsilon Decay**: Slowly reduces exploration over time

### Key Functions:

- `choose_action(state, available_moves)`:  
  - Uses **heuristic** first (win/block).  
  - Else, uses **epsilon-greedy Q-Learning**:
    - With probability ε → random move (exploration)  
    - With probability 1-ε → move with max Q-value (exploitation)
- `learn(state, action, reward, next_state, done, available_moves)`:  
  - Updates Q-values using **Temporal Difference (TD) learning**:

\[
Q(s, a) \gets Q(s, a) + \alpha \cdot \big(r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)\big)
\]

- `heuristic_move(board, player)`:  
  - If agent can win in one move → take it  
  - If opponent can win in one move → block it  

---

## Training Logic

- Training happens in `train.py`
- Each **episode** = one full game
- **Steps in an episode**:
  1. Reset board
  2. Agent selects a move
  3. Execute move → get reward and next state
  4. Opponent plays random move
  5. Update Q-table
  6. Repeat until game ends
- Print progress every N episodes
- After training, agent can play optimally using the learned Q-table

---

## Heuristic Rules

Heuristics are simple **immediate strategies** that improve performance:

1. **Win**: If a move leads to an immediate win → take it
2. **Block**: If opponent can win in next move → block it

These rules are applied **before Q-Learning decisions**, ensuring the agent doesn’t miss obvious moves early in training.

---

## Gameplay

- `play.py` allows **human vs AI** play.
- Human inputs move (0–8)
- Agent moves automatically using either heuristic or learned Q-table
- Safe input handling: prevents moves in occupied cells
- Board prints after each move

---

## Q-Learning Equation

Mathematically, Q-Learning updates each state-action pair as:

\[
Q(s, a) \gets Q(s, a) + \alpha \cdot \big(\text{reward} + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)\big)
\]

Where:  
- \(s\) = current state  
- \(a\) = action taken  
- \(\alpha\) = learning rate  
- \(r\) = reward for taking action  
- \(\gamma\) = discount factor  
- \(s'\) = next state after action  
- \(a'\) = possible actions in next state  

**Example**:

| State        | Action | Reward | Next State       | Max Q(next) | Updated Q |
|--------------|--------|--------|-----------------|-------------|-----------|
| Empty board  | 4      | 0      | after agent move | 0           | 0         |
| Partial board| 8      | 0      | after opponent   | 0           | 0         |
| Winning move | 1      | 1      | done             | N/A         | 0 + 0.1*(1-0)=0.1 |

---


