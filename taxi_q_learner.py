import csv
import numpy as np
import gymnasium as gym
import os
from rich.console import Console
from rich.table import Table

class TaxiQLearner:
    """
    Tabular Q-Learning agent for the Taxi-v3 environment.

    Parameters
    ----------
    learning_rate : float
        Learning rate (α), controls how much new information overrides old.
    discount_factor : float
        Discount factor (γ), determines the importance of future rewards.
    initial_exploration : float
        Initial exploration rate for ε-greedy policy.
    min_epsilon : float
        Minimum allowable value of ε after decay.
    decay_factor : float
        Multiplicative decay factor applied to ε after each episode past threshold.
    decay_threshold : int
        Number of episodes after which ε starts decaying.
    max_steps : int
        Maximum steps allowed per episode.
    modified_env : gym.Wrapper, optional
        A custom wrapper (e.g., TaxiQDomainModifier) to modify the reward structure.
        If None, the default environment will be used.
    """
    
    def __init__(self, learning_rate=0.8, discount_factor=0.95, initial_exploration=1.0, min_epsilon=0.01, decay_factor=0.999, decay_threshold=2000, max_steps=100, modified_env=None):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_exploration
        self.min_epsilon = min_epsilon
        self.decay_factor = decay_factor
        self.decay_threshold = decay_threshold

        self.max_steps = max_steps

        self.episodes_trained = 0
        
        # Wrap the environment with the domain modifier if provided
        self.env = modified_env if modified_env else gym.make('Taxi-v3')
        self.using_modified_env = modified_env is not None
        
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))

    def summary(self):
        """
        Pretty print the current model parameters and hyperparameters using the rich library.
        """
        table = Table(title="Taxi Q-Learning Model Parameters")

        # Add columns for the parameter names and their values
        table.add_column("Parameter", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="sky_blue1")

        # Add rows for each parameter
        table.add_row("Learning Rate (α)", f"{self.alpha:.3f}")
        table.add_row("Discount Factor (γ)", f"{self.gamma:.3f}")
        table.add_row("Initial Exploration (ε)", f"{self.epsilon:.3f}")
        table.add_row("Minimum Exploration (min ε)", f"{self.min_epsilon:.3f}")
        table.add_row("Decay Factor", f"{self.decay_factor:.3f}")
        table.add_row("Decay Threshold", f"{self.decay_threshold}")
        table.add_row("Max Steps per Episode", f"{self.max_steps}")
        table.add_row("Using Modified Environment", str(self.using_modified_env))
        table.add_row("Episodes Trained", str(self.episodes_trained))
        
        # Print the table using the rich library
        console = Console()
        console.print(table)

    def update_q_table(self, current_state, action_taken, reward_received, next_state):
        """
        Update Q-value using the Bellman equation.

        Parameters
        ----------
        current_state : int
            Current state index.
        action_taken : int
            Action taken in the current state.
        reward_received : float
            Reward received after taking the action.
        next_state : int
            Resulting state after taking the action.
        """
        sample = reward_received + self.gamma * np.max(self.Q[next_state])
        self.Q[current_state, action_taken] = (1 - self.alpha) * self.Q[current_state, action_taken] + self.alpha * sample

    def log_to_csv(self, filename, total_reward):
        """
        Log episode data to a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV log file.
        total_reward : float
            Total reward obtained in the episode.
        """

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.episodes_trained, total_reward])

    def train(self, log_to_csv=True, log_path='training_log.csv', n_episodes=5000, verbose= True):
        """
        Train the Q-learning agent.

        Parameters
        ----------
        log_to_csv : bool
            If True, log training data to a CSV file.
        log_path : str
            Path to the training log CSV file.
        n_episodes : int
            Number of episodes to train the agent.
        verbose : bool
            If True, print completion message after training.
        """
        if log_to_csv and self.episodes_trained == 0: # Resets the log file if it exists or creates a new one
            with open(log_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['episode', 'total_reward'])

        for episode in range(1, n_episodes + 1):
            state, _ = self.env.reset()
            self.episodes_trained += 1
            total_reward = 0

            for _ in range(self.max_steps):
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward

                if done:
                    break

            if episode > self.decay_threshold: # Start decaying epsilon after threshold
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor)

            if log_to_csv:
                self.log_to_csv(log_path, total_reward)

        self.env.close()

        if verbose:
            print(f"Training completed after {n_episodes} episodes.")
            if log_to_csv:
                print(f"Training log saved to {log_path}")

    def save_q_table(self, filename='q_table.csv', verbose=True):
        """
        Normalize and save the Q-table to a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file where Q-table will be saved.
        verbose : bool
            If True, print completion message after saving.
        """
      
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['state'] + [f'action_{i}' for i in range(self.n_actions)])
            for state_idx in range(self.n_states):
                writer.writerow([state_idx] + self.Q[state_idx].tolist())
        
        if verbose:
            print(f"Q-table saved to {filename}")

    def load_q_table(self, filename='q_table.csv', verbose=True):
        """
        Load Q-table from a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file from which Q-table will be loaded.
        verbose : bool
            If True, print completion message after loading.
        """
        self.Q = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=range(1, self.n_actions + 1))
        
        if verbose:
            print(f"Q-table loaded from {filename}")

    def run(self, render_mode: str = None, num_episodes: int = 1, verbose: int = 1) -> float:
        """
        Run the trained agent in the environment.

        Parameters
        ----------
        render_mode : str
            Mode for rendering the environment. Options are 'human' or 'rgb_array'.
        num_episodes : int
            The number of episodes to run.
        verbose : int
            Verbosity level:
            - 0: No print.
            - 1: Print average reward after all episodes.
            - 2: Print reward for each episode + average reward at the end.
        
        Returns
        -------
        float
            Average reward over the specified number of episodes.
        """
        env = gym.make('Taxi-v3', render_mode=render_mode)  # Create the environment once
        total_rewards = []  # To store rewards for each episode

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = np.argmax(self.Q[state])
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if render_mode == 'human':
                    env.render()

            total_rewards.append(total_reward)

            if verbose == 2:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        env.close()  # Close the environment after all episodes are finished
        avg_reward = np.mean(total_rewards)

        if verbose >= 1:
            print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
        
        return avg_reward