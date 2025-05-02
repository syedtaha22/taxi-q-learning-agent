import numpy as np
import gymnasium as gym
from taxi_q_learner import TaxiQLearner

if __name__ == "__main__":
    agent = TaxiQLearner()
    agent.train('training_log.csv')
    agent.save_q_table('q_table.csv')

    agent.run(num_episodes=1000, render_mode='human', verbose=1)

    # try:
    #     env = gym.make('Taxi-v3', render_mode='human')
    #     print("Training complete. Press Ctrl+C to exit.")
    #     runs = 0 # For calculating average reward over runs
    #     total_reward = 0 # For calculating average reward over runs

    #     while True:
    #         state, _ = env.reset()
    #         done = False
    #         runs += 1
    #         while not done:
    #             action = np.argmax(agent.Q[state])
    #             state, reward, terminated, truncated, _ = env.step(action)
    #             done = terminated or truncated
    #             total_reward += reward
    #         print(f"Run {runs}: Average Reward: {total_reward/runs:.2f}")
    # except KeyboardInterrupt:
    #     print("\nDemo terminated.")
