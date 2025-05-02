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
    alpha : float
        Learning rate (α), controls how much new information overrides old.
    gamma : float
        Discount factor (γ), determines the importance of future rewards.
    epsilon : float
        Initial exploration rate for ε-greedy policy.
    min_epsilon : float
        Minimum allowable value of ε after decay.
    decay_factor : float
        Multiplicative decay factor applied to ε after each episode past threshold.
    decay_threshold : int
        Number of episodes after which ε starts decaying.
    n_episodes : int
        Number of training episodes.
    max_steps : int
        Maximum steps allowed per episode.
    """
    
    def __init__(self, alpha=0.8, gamma=0.95, epsilon=1.0, min_epsilon=0.01, decay_factor=0.999, decay_threshold=2000, n_episodes=5000, max_steps=100):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_factor = decay_factor
        self.decay_threshold = decay_threshold

        self.n_episodes = n_episodes
        self.max_steps = max_steps
        
        self.env = gym.make('Taxi-v3')
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
        table.add_row("Number of Episodes", f"{self.n_episodes}")
        table.add_row("Max Steps per Episode", f"{self.max_steps}")
        
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

    def log_to_csv(self, filename, episode, total_reward):
        """
        Log episode data to a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV log file.
        episode : int
            Current episode number.
        total_reward : float
            Total reward obtained in the episode.
        """

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, total_reward])

    def train(self, log_to_csv=True, log_path='training_log.csv', verbose= True):
        """
        Train the Q-learning agent.

        Parameters
        ----------
        log_to_csv : bool
            If True, log training data to a CSV file.
        log_path : str
            Path to the training log CSV file.
        verbose : bool
            If True, print completion message after training.
        """
        if log_to_csv:
            with open(log_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['episode', 'total_reward'])

        for episode in range(1, self.n_episodes + 1):
            state, _ = self.env.reset()
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
                self.log_to_csv(log_path, episode, total_reward)

        self.env.close()

        if verbose:
            print(f"Training completed after {self.n_episodes} episodes.")
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
        q_min = np.min(self.Q)
        q_max = np.max(self.Q)
        if q_max != q_min:
            self.Q = (self.Q - q_min) / (q_max - q_min)

        # Round Q-values to 3 decimal places
        self.Q = np.round(self.Q, 3)
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['state'] + [f'action_{i}' for i in range(self.n_actions)])
            for state_idx in range(self.n_states):
                writer.writerow([state_idx] + self.Q[state_idx].tolist())
        
        if verbose:
            print(f"Q-table saved to {filename}")

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
