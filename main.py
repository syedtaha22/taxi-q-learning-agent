import numpy as np
import gymnasium as gym
from taxi_q_learner import TaxiQLearner

# dir_name = 'hyperparameter_tuning'
# dir_name = 'baseline'
dir_name = 'modified_environment'

if __name__ == "__main__":
    agent = TaxiQLearner()
    agent.load_q_table(f'{dir_name}/q_table.csv', verbose=True)

    try:
        env = gym.make('Taxi-v3', render_mode='human')
        print("Training complete. Press Ctrl+C to exit.")
        runs = 0 # For calculating average reward over runs
        total_reward = 0 # For calculating average reward over runs

        while True:
            state, _ = env.reset()
            done = False
            runs += 1
            while not done:
                action = np.argmax(agent.Q[state])
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
            print(f"Run {runs}: Average Reward: {total_reward/runs:.2f}")
    except KeyboardInterrupt:
        print("\nDemo terminated.")
