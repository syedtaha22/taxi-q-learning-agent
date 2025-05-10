import gymnasium as gym
from gymnasium.core import Wrapper

class TaxiQDomainModifier(Wrapper):
    """
    A wrapper for modifying the reward structure in the Taxi-v3 environment for Q-learning.

    This wrapper allows custom modifications to the rewards for specific actions
    taken in the environment, such as penalizing each step, assigning penalties
    for illegal moves, rewarding successful drop-offs, and shaping rewards using
    changes in Manhattan distance between the taxi and the passenger/destination.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    step_penalty : int, optional
        The penalty for each step taken in the environment. Default is -1.
    illegal_move_penalty : int, optional
        The penalty for illegal moves (such as picking up or dropping off in an invalid location).
        Default is -10.
    dropoff_reward : int, optional
        The reward for successfully completing a drop-off. Default is 20.
    manhattan_weight : float, optional
        A shaping term added as `-weight * delta`, where delta is the change
        in Manhattan distance to the goal. Default is 0 (disabled).
    euclidean_weight : float, optional
        A shaping term added as `-weight * delta`, where delta is the change
        in Euclidean distance to the goal. Default is 0 (disabled).
    """

    def __init__(self, env, step_penalty=-1, illegal_move_penalty=-10,
                 dropoff_reward=20, manhattan_weight=0.0, euclidean_weight=0.0):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.illegal_move_penalty = illegal_move_penalty
        self.dropoff_reward = dropoff_reward
        self.manhattan_weight = manhattan_weight
        self.euclidean_weight = euclidean_weight

        self.prev_goal = None
        self.prev_distance = None

    def step(self, action):
        """
        Take a step in the environment and modify the reward based on the custom structure.

        Parameters
        ----------
        action : int
            The action to take in the environment.

        Returns
        -------
        state : int
            The encoded state after taking the action.
        reward : float
            The modified reward after applying step penalty, illegal move penalty, 
            drop-off reward, and change in Manhattan distance shaping.
        terminated : bool
            Whether the episode has ended due to reaching the goal or a terminal state.
        truncated : bool
            Whether the episode was truncated due to time limits or max steps.
        info : dict
            Extra information returned by the environment, typically used for debugging.
        """
        state, reward, terminated, truncated, info = self.env.step(action)

        # Decode state into taxi_row, taxi_col, passenger_location, destination
        taxi_row, taxi_col, pass_loc, dest_idx = self.env.unwrapped.decode(state)

        locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B

        # Determine current goal
        if pass_loc < 4:
            goal = locations[pass_loc]
        else:
            goal = locations[dest_idx]

        current_distance = abs(taxi_row - goal[0]) + abs(taxi_col - goal[1])

        # Reward modification
        if reward == -1:
            reward = self.step_penalty
        elif reward == -10:
            reward = self.illegal_move_penalty
        elif reward == 20:
            reward = self.dropoff_reward

        # Compute shaping reward if goal hasn't changed
        if self.prev_goal == goal and self.prev_distance is not None:
            manhattan_distance = abs(taxi_row - goal[0]) + abs(taxi_col - goal[1])
            euclidean_distance = ((taxi_row - goal[0]) ** 2 + (taxi_col - goal[1]) ** 2) ** 0.5
            reward += - self.manhattan_weight * manhattan_distance 
            reward += - self.euclidean_weight * euclidean_distance
            # delta = self.prev_distance - current_distance
            # reward += min(0, self.manhattan_weight * delta)  # negative if further away
            # reward += self.manhattan_weight * delta  # positive if closer
        else:
            # Reset memory on pickup/dropoff
            self.prev_goal = goal

        self.prev_distance = current_distance

        return state, reward, terminated, truncated, info
