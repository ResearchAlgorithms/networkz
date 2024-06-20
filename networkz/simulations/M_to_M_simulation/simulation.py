"""
Comparing the performance of the custom implementation of the Kuhn-Munkres with Backtracking algorithm with the Munkres implementation.
"""

import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from fairpyx.utils.graph_utils import many_to_many_matching
from networkz.algorithms.bipartite.many_to_many_assignment import kuhn_munkers_backtracking
import experiments_csv as ec

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

def generate_random_test_case(num_agents, num_tasks, max_ability, max_task_range, max_performance_value):
    """
    Generate a random test case for the many-to-many assignment problem.
    
    Parameters:
    `num_agents (int)`: Number of agents.
    `num_tasks (int)`: Number of tasks.
    `max_ability (int)`: Maximum ability value for an agent.
    `max_task_range (int)`: Maximum range value for a task.
    `max_performance_value (int)`: Maximum performance value for the matrix.

    Returns:
    `tuple`: A tuple containing the ability_agent_vector, task_range_vector, and performance_matrix.
    """
    ability_agent_vector = np.random.randint(1, max_ability + 1, size=num_agents)
    task_range_vector = np.random.randint(1, max_task_range + 1, size=num_tasks)
    
    # Ensure the cardinality constraint is satisfied
    if ability_agent_vector.sum() < task_range_vector.sum():
        # Scale down task range vector to satisfy the constraint
        factor = ability_agent_vector.sum() / task_range_vector.sum()
        task_range_vector = np.floor(task_range_vector * factor).astype(int)
    
    if task_range_vector.sum() == 0:
        task_range_vector[np.random.randint(0, num_tasks)] = 1

    performance_matrix = np.random.randint(1, max_performance_value + 1, size=(num_agents, num_tasks))
    
    return ability_agent_vector, task_range_vector, performance_matrix

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
    ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(50, 50, 3, 3, 100)

    # Convert inputs for fairxpy function
    item_capacities = {f"item_{i}": task_range_vector[i] for i in range(len(task_range_vector))}
    agent_capacities = {f"agent_{i}": ability_agent_vector[i] for i in range(len(ability_agent_vector))}
    valuations = {f"agent_{i}": {f"item_{j}": performance_matrix[i][j] for j in range(len(performance_matrix[i]))} for i in range(len(performance_matrix))}
    
    # Measure time for custom implementation
    munkers_with_backtracking_time,_ = measure_time(kuhn_munkers_backtracking, performance_matrix, ability_agent_vector, task_range_vector)

    # Measure time for fairxpy implementation
    fairxpy_time, _ = measure_time(many_to_many_matching, item_capacities, agent_capacities, valuations)


    # Return the results in the format expected by experiments_csv
    return {
        'size': size,
        'munkers_with_backtracking_time': munkers_with_backtracking_time,
        'fairxpy_many_to_many_matching_time': fairxpy_time
    }

input_ranges = {
    'size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
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
plt.plot(df['size'], df['munkers_with_backtracking_time'], label='Munkers with Backtracking')
plt.plot(df['size'], df['fairxpy_many_to_many_matching_time'], label='Fairxpy Many-to-Many Matching')
plt.xlabel('Size')
plt.ylabel('Time (seconds)')
plt.legend()
plot_path = 'results/performance_comparison_plot.png'
plot_full_path = os.path.join(os.getcwd(), plot_path)
print(plot_full_path)
plt.savefig(plot_path)
plt.show()
print(f"Plot saved at {plot_path}")