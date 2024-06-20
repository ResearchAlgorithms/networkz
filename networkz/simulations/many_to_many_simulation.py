"""
Comparing the performance of the custom implementation of the Kuhn-Munkres with Backtracking algorithm with the Munkres implementation.
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from munkres import Munkres
from networkz.algorithms.bipartite.many_to_many_assignment import kuhn_munkers_backtracking
import experiments_csv as ec

def munkres_assignment_method(cost_matrix: np.array):
    """
    Define the Munkers wrapper.

    Parameters
    ----------
    - `cost_matrix` : np.array

    Returns
    ----------
    - `total_cost` : float
    """
    m = Munkres()
    indexes = m.compute(cost_matrix)
    total_cost = sum(cost_matrix[row][column] for row, column in indexes)
    return total_cost

def measure_time(func, *args) -> tuple[float, any]:
    """
    Measure the execution time of a function

    Parameters
    ----------
    - `func` : function
    - `cost_matrix` : np.array

    Returns
    ----------
    - `time` : float
    - `result` : any
    """
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return end_time - start_time, result

def generate_random_cost_matrix(size, max_value = 100) -> np.array:
    """
    Generate a random cost matrix of the given size.

    Parameters
    ----------
    - `size` : int
    - `max_value` : int

    Returns
    ----------
    - `cost_matrix` : np.array
    """
    return np.random.randint(1, max_value + 1, size=(size, size))

def run_experiment(size: int) -> dict:
    """
    Run the experiment for the given size.

    Parameters
    ----------
    - `size` : int

    Returns
    ----------
    - `dict` : dict
    """
    # Generate a random cost matrix of the given size
    cost_matrix = generate_random_cost_matrix(size)
    
    # Initialize ability_agent_vector and task_range_vector with ones
    ability_agent_vector = np.ones(size, dtype=int)
    task_range_vector = np.ones(size, dtype=int)
    
    # Measure time for custom implementation
    munkers_with_backtracking_time,_ = measure_time(kuhn_munkers_backtracking, cost_matrix, ability_agent_vector, task_range_vector)
    
    # Measure time for Munkres implementation
    munkres_without_backtracking_time,_ = measure_time(munkres_assignment_method, cost_matrix)
    
    # Return the results in the format expected by experiments_csv
    return {
        'size': size,
        'munkers_with_backtracking_time': munkers_with_backtracking_time,
        'munkres_without_backtracking_time': munkres_without_backtracking_time
    }

input_ranges = {
    'size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 200, 250, 300, 350, 400, 450, 500]
}

# Initialize the experiment
ex = ec.Experiment(results_folder='results', results_filename='assignment_experiment_results.csv', backup_folder='results_backup')

# Run the experiment
ex.clear_previous_results()  # Uncomment if you want to restart all experiments from scratch
ex.run(run_experiment, input_ranges)

# Load the results
df = pd.read_csv('results/assignment_experiment_results.csv')

print(df)

# Plot the results
plt.plot(df['size'], df['munkers_with_backtracking_time'], label='Munkers With Backtracking Method')
plt.plot(df['size'], df['munkres_without_backtracking_time'], label='Munkres Without Backtracking Method')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds)')
plt.legend()
plot_path = 'results/performance_comparison_plot.png'
plt.savefig(plot_path)
plt.show()
print(f"Plot saved at {plot_path}")