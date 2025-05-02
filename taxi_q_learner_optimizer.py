import os
import csv
from itertools import product
from taxi_q_learner import TaxiQLearner


class TaxiQLearnerOptimizer:
    """
    Exhaustive grid search optimizer for TaxiQLearner hyperparameters.

    Parameters
    ----------
    learning_rates : list of float
        List of learning rates to test during optimization.
    discount_factors : list of float
        List of discount factors (gamma) to test.
    exploration_rates : list of float
        List of exploration rates (epsilon) to test.
    decay_factors : list of float
        List of decay factors to test.
    decay_thresholds : list of int
        List of decay thresholds to test.
    output_dir : str, optional, default="hyperparameter_tuning"
        Base directory to save logs, Q-tables, and results.
    """

    def __init__(
        self,
        learning_rates,
        discount_factors,
        exploration_rates,
        decay_factors,
        decay_thresholds,
        output_dir="hyperparameter_tuning"
    ):
        self.learning_rates = learning_rates
        self.discount_factors = discount_factors
        self.exploration_rates = exploration_rates
        self.decay_factors = decay_factors
        self.decay_thresholds = decay_thresholds

        self.logs_dir = os.path.join(output_dir, "training_logs")
        self.qtables_dir = os.path.join(output_dir, "q_tables")
        self.results_csv = os.path.join(output_dir, "grid_search_results.csv")

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.qtables_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize CSV file with header
        with open(self.results_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ 
                "LearningRate", "DiscountFactor", "ExplorationRate",
                "DecayFactor", "DecayThreshold", "AverageReward"
            ])

    def optimize(self, verbose=False, save_artifacts=True):
        """
        Perform grid search over hyperparameter combinations.

        Parameters
        ----------
        verbose : bool, optional, default=False
            Whether to print results as they are computed.
        save_artifacts : bool, optional, default=True
            Whether to save training logs and Q-tables for each run.

        Notes
        -----
        For each combination of hyperparameters, the agent is trained, the Q-table 
        is saved (if `save_artifacts` is True), and the average reward is logged 
        in the specified CSV file.
        """
        combinations = product(
            self.learning_rates,
            self.discount_factors,
            self.exploration_rates,
            self.decay_factors,
            self.decay_thresholds
        )

        for lr, gamma, epsilon, decay, threshold in combinations:
            file_suffix = (
                f"learning_rate_{lr}_"
                f"discount_factor_{gamma}_"
                f"exploration_rate_{epsilon}_"
                f"decay_factor_{decay}_"
                f"decay_threshold_{threshold}"
            )

            log_file = os.path.join(self.logs_dir, f"{file_suffix}.csv")
            q_table_file = os.path.join(self.qtables_dir, f"{file_suffix}.csv")

            agent = TaxiQLearner(
                alpha=lr,
                gamma=gamma,
                epsilon=epsilon,
                decay_factor=decay,
                decay_threshold=threshold
            )
            
            agent.train(log_to_csv=save_artifacts, log_path=log_file, verbose=False)

            if save_artifacts:
                agent.save_q_table(q_table_file, verbose=False)

            avg_reward = agent.run(num_episodes=100, verbose=0)

            # Append result to CSV immediately
            with open(self.results_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([lr, gamma, epsilon, decay, threshold, avg_reward])

            if verbose:
                print(f"LearningRate: {lr}, DiscountFactor: {gamma}, ExplorationRate: {epsilon}, "
                      f"DecayFactor: {decay}, DecayThreshold: {threshold}, AvgReward: {avg_reward}")
