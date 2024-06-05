import unittest
import numpy as np
from networkz.algorithms.bipartite.many_to_many_assignment import kuhn_munkers_backtracking as kmb

class test_many_to_many_assignment(unittest.TestCase):
    def test_example_1(self):
        ability_agent_vector = np.array([1, 1, 1])
        task_range_vector = np.array([1, 1, 1])
        matrix = np.array([[30, 25, 10],[15, 10, 20],[25, 20, 15]])
        output = kmb(matrix=matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        expected_output = {0: [2], 1: [1], 2: [0]}
        self.assertEqual(output, expected_output)

    def test_example_2(self):
        matrix = np.array([[40, 60, 15],[25, 30, 45],[55, 30, 25]])
        ability_agent_vector = np.array([1, 1, 1])
        task_range_vector = np.array([1, 1, 1])
        output = kmb(matrix=matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        expected_output = {0: [2], 1: [0], 2: [1]}
        self.assertEqual(output, expected_output)
    
    def test_example_3(self):
        matrix = np.array([[3, 0, 1, 2],[2, 3, 0, 1],[3, 0, 1, 2],[1, 0, 2, 3]])
        ability_agent_vector = np.array([2,2,2,2])
        task_range_vector = np.array([2,2,2,2])
        output = kmb(matrix=matrix, agentVector=ability_agent_vector, taskRangeVector=task_range_vector)
        expected_output = {0: [3, 0], 1: [2, 3], 2: [1, 2], 3: [0, 1]}
        self.assertEqual(output, expected_output)