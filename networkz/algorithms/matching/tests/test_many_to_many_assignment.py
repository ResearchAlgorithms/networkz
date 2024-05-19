import pytest
import numpy as np
import networkz as nx
import networkz.algorithms.matching.many_to_many_assignment as mma

def arrays_are_equal(arr1: np.array, arr2: np.array):
    assert arr1.shape == arr2.shape, f"Shape mismatch: {arr1.shape} != {arr2.shape}"
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[1]):
            assert arr1[i, j] == arr2[i, j], f"element mismatch at ({i}, {j}): {arr1[i, j]} != {arr2[i, j]}"

def check_constraints(output, ability_agent_vector, task_range_vector):
    assert set(output.flatten()).issubset({0, 1}), "Output matrix contains values other than 0 and 1"
    for i in range(output.shape[0]):
        assert sum(output[i, :]) <= ability_agent_vector[i], f"Agent {i} assigned too many tasks"
    for j in range(output.shape[1]):
        assert sum(output[:, j]) <= task_range_vector[j], f"Task {j} assigned to too many agents"

def generate_random_test_case(seed, size_agents, size_tasks, ability_max, task_max):
    np.random.seed(seed)
    ability_agent_vector = np.random.randint(1, ability_max + 1, size=size_agents)
    task_range_vector = np.random.randint(1, task_max + 1, size=size_tasks)
    performance_matrix = np.random.randint(0, 100, size=(size_agents, size_tasks))
    return ability_agent_vector, task_range_vector, performance_matrix

class TestManyToManyAssignment:
    def test_example_1():
        ability_agent_vector = np.array([1, 2, 1, 1])
        task_range_vector = np.array([1, 1, 1, 1, 1])
        performance_matrix = np.array([[49, 45, 39, 15, 16], 
                                    [5, 30, 85, 22, 78], 
                                    [61, 16, 71, 59, 20], 
                                    [44, 79, 1, 48, 22]])
        expected_output = np.array([[0, 0, 0, 0, 1], 
                                    [1, 0, 0, 1, 0], 
                                    [0, 1, 0, 0, 0], 
                                    [0, 0, 1, 0, 0]])
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        arrays_are_equal(output, expected_output)

    def test_example_2():
        ability_agent_vector = np.array([1, 1, 1])
        task_range_vector = np.array([1, 1, 1])
        performance_matrix = np.array([[40, 60, 15], 
                                    [25, 30, 45], 
                                    [55, 30, 25]])
        expected_output = np.array([[0, 0, 1], 
                                    [1, 0, 0], 
                                    [0, 1, 0]])
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        arrays_are_equal(output, expected_output)

    def test_example_3():
        ability_agent_vector = np.array([1, 1, 1])
        task_range_vector = np.array([1, 1, 1])
        performance_matrix = np.array([[30, 25, 10], 
                                    [15, 10, 20], 
                                    [25, 20, 15]])
        expected_output = np.array([[0, 0, 1], 
                                    [0, 1, 0], 
                                    [1, 0, 0]])
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        arrays_are_equal(output, expected_output)

    def test_example_4():
        ability_agent_vector = np.array([2, 2, 1, 3])
        task_range_vector = np.array([1, 2, 3, 1, 1])
        performance_matrix = np.array([[8, 6, 7, 9, 5], 
                                    [6, 7, 8, 6, 7], 
                                    [7, 8, 5, 6, 8], 
                                    [7, 6, 9, 7, 5]])
        with pytest.raises(ValueError, match="The Cordinality Constraint is not satisfied"):
            mma(ability_agent_vector, task_range_vector, performance_matrix)

    def test_example_5():
        ability_agent_vector = np.array([1, 1, 1])
        task_range_vector = np.array([1, 1, 1])
        performance_matrix = np.array([[-30, 25, -1], 
                                    [-7, 10, 2], 
                                    [25, -4, -3]])
        with pytest.raises(ValueError, match="The performance matrix has values less then 0"):
            mma(ability_agent_vector, task_range_vector, performance_matrix)

    def test_empty_input():
        ability_agent_vector = np.array([])
        task_range_vector = np.array([])
        performance_matrix = np.array([])
        with pytest.raises(ValueError, match="Empty input"):
            mma(ability_agent_vector, task_range_vector, performance_matrix)

    def text_example_6():
        ability_agent_vector = np.array([1, 2, 1])
        task_range_vector = np.array([1, 1, 1, 1])
        performance_matrix = np.array([[3, 4, 5, 2], 
                                    [5, 2, 3, 4], 
                                    [1, 6, 2, 3]])
        expected_output = np.array([[0, 0, 0, 1], 
                                    [0, 1, 1, 0], 
                                    [0, 0, 1, 0]])
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        arrays_are_equal(output, expected_output)

    def test_large_example_1():
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(0, 50, 50, 3, 3)
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        check_constraints(output, ability_agent_vector, task_range_vector)

    def test_large_example_2():
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(1, 100, 100, 3, 3)
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        check_constraints(output, ability_agent_vector, task_range_vector)

    def test_random_example_1():
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(2, 10, 10, 3, 3)
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        check_constraints(output, ability_agent_vector, task_range_vector)

    def test_random_example_2():
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(3, 20, 20, 4, 4)
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        check_constraints(output, ability_agent_vector, task_range_vector)

    def test_random_example_3():
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(4, 30, 30, 5, 5)
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        check_constraints(output, ability_agent_vector, task_range_vector)

    def test_random_example_4():
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(5, 40, 40, 3, 3)
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        check_constraints(output, ability_agent_vector, task_range_vector)

    def test_random_example_5():
        ability_agent_vector, task_range_vector, performance_matrix = generate_random_test_case(6, 50, 50, 4, 4)
        output = mma(ability_agent_vector, task_range_vector, performance_matrix)
        check_constraints(output, ability_agent_vector, task_range_vector)