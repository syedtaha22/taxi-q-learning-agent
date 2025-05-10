import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

class TaxiQVisualizer:
    """
    A class for visualizing Q-values of the Taxi-v3 environment using triangle heatmaps.

    Parameters
    ----------
    q_table_path : str
        Path to the CSV file containing the Q-table.
    """

    def __init__(self, q_table_path):
        self.Q = self._load_q_table(q_table_path)
        self.loc_map = {
            0: (0, 0),  # R
            1: (0, 4),  # G
            2: (4, 0),  # Y
            3: (4, 3)   # B
        }

    def _load_q_table(self, path):
        """
        Load the Q-table from a CSV file.

        Parameters
        ----------
        path : str
            Path to the Q-table CSV file.

        Returns
        -------
        numpy.ndarray
            Array of shape (500, 6) containing Q-values for each state-action pair.
        """
        df = pd.read_csv(path)
        return df[[f'action_{i}' for i in range(6)]].to_numpy()

    def decode_state(self, state):
        """
        Decode a flat state index into its components.

        Parameters
        ----------
        state : int
            Encoded environment state.

        Returns
        -------
        tuple
            A tuple (taxi_row, taxi_col, passenger_loc, destination).
        """
        destination = state % 4
        passenger_loc = (state // 4) % 5
        taxi_col = (state // 20) % 5
        taxi_row = state // 100
        return taxi_row, taxi_col, passenger_loc, destination

    def _get_state_index(self, row, col, passenger_loc, destination):
        """
        Compute the flat state index from environment components.

        Parameters
        ----------
        row : int
            Taxi's row position.
        col : int
            Taxi's column position.
        passenger_loc : int
            Passenger location (0–4).
        destination : int
            Destination index (0–3).

        Returns
        -------
        int
            Encoded state index.
        """
        return (((row * 5 + col) * 5) + passenger_loc) * 4 + destination

    def _normalize_q_values(self, q_vals):
        """
        Normalize Q-values to the range [-1, 1], preserving sign.

        Parameters
        ----------
        q_vals : array-like
            Q-values for a given state.

        Returns
        -------
        list of float
            Normalized Q-values.
        """
        max_abs = np.max(np.abs(q_vals))
        if max_abs == 0:
            return [0.0 for _ in q_vals]
        return [q / max_abs for q in q_vals]

    def _draw_grid(self, ax):
        """
        Draw a 5x5 grid on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes to draw on.
        """
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_xticks(np.arange(5) + 0.5)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_xticklabels(np.arange(5))
        ax.set_yticklabels(np.arange(5))
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid(False)
        ax.set_aspect('equal')

        for i in range(6):
            ax.axhline(i, color='black', linewidth=1)
            ax.axvline(i, color='black', linewidth=1)

    def _draw_cell(self, ax, row, col, q_vals):
        """
        Draw a cell at (row, col) with triangles representing Q-values.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw the cell on.
        row : int
            Row index of the cell.
        col : int
            Column index of the cell.
        q_vals : array-like
            Q-values for the state.
        """
        norm_vals = self._normalize_q_values(q_vals)
        x, y = col, row
        center = (x + 0.5, y + 0.5)
        corners = {
            'tl': (x, y + 1),
            'tr': (x + 1, y + 1),
            'br': (x + 1, y),
            'bl': (x, y)
        }

        triangles = {
            0: [corners['tl'], center, corners['tr']],  # North
            1: [corners['bl'], center, corners['br']],  # South
            2: [corners['br'], center, corners['tr']],  # East
            3: [corners['bl'], center, corners['tl']]   # West
        }

        for a in range(4):
            shade = norm_vals[a]

            if shade > 0:
                color = plt.cm.Greens(shade)
            elif shade < 0:
                color = plt.cm.Reds(-shade)
            else:
                color = (1, 1, 1, 0.5) # White for zero Q-value
            polygon = Polygon(triangles[a], facecolor=color, edgecolor='black', alpha=0.5, lw=0.3)
            ax.add_patch(polygon)

            cx = np.mean([p[0] for p in triangles[a]])
            cy = np.mean([p[1] for p in triangles[a]])
            ax.text(cx, cy, f'{norm_vals[a]:.2f}', ha='center', va='center', fontsize=8, alpha=0.5)

    def _annotate_cell_entities(self, ax, taxi_row, taxi_col, passenger_loc, destination):
        """
        Annotate taxi, passenger, and destination positions on the grid.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes on which to annotate.
        taxi_row : int
            Taxi's row index.
        taxi_col : int
            Taxi's column index.
        passenger_loc : int
            Passenger location index.
        destination : int
            Destination location index.

        Returns
        -------
        tuple
            Coordinates of passenger (px, py) and destination (dx, dy).
        """
        ax.text(taxi_col + 0.5, taxi_row + 0.5, 'T', ha='center', va='center', fontsize=16, color='blue')

        if passenger_loc < 4:
            py, px = self.loc_map[passenger_loc]
            ax.text(px + 0.5, py + 0.5, 'P', ha='center', va='center', fontsize=16, color='red')
        else:
            px = py = -1

        dy, dx = self.loc_map[destination]
        ax.text(dx + 0.5, dy + 0.5, 'D', ha='center', va='center', fontsize=16, color='red')

        return px, py, dx, dy

    def plot_state(self, taxi_row, taxi_col, passenger_loc, destination):
        """
        Plot the Q-values for a specific environment state configuration.

        Parameters
        ----------
        taxi_row : int
            Row index of the taxi.
        taxi_col : int
            Column index of the taxi.
        passenger_loc : int
            Passenger location index (0–4).
        destination : int
            Destination location index (0–3).
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_grid(ax)

        for row in range(5):
            for col in range(5):
                state_idx = self._get_state_index(row, col, passenger_loc, destination)
                q_vals = self.Q[state_idx, :4]
                self._draw_cell(ax, row, col, q_vals)

        px, py, dx, dy = self._annotate_cell_entities(ax, taxi_row, taxi_col, passenger_loc, destination)
        ax.set_title(f'Taxi at ({taxi_col},{taxi_row}) | P=({px},{py}) | D=({dx},{dy})', fontsize=10)
        plt.show()

    def run_random_state(self):
        """
        Select a random state and plot its corresponding Q-value visualization.
        """
        state = random.randint(0, self.Q.shape[0] - 1)
        taxi_row, taxi_col, passenger_loc, destination = self.decode_state(state)
        self.plot_state(taxi_row, taxi_col, passenger_loc, destination)

# def main():
#     visualizer = TaxiQVisualizer('q_table.csv')
#     visualizer.run_random_state()

# if __name__ == '__main__':
#     main()
