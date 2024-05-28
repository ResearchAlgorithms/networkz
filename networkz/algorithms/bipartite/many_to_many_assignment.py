"""
	An implementation of the algorithms in:
 
	"Solving the Many to Many assignment problem by improving
       the Kuhn–Munkres algorithm with backtracking", by:
       Haibin Zhu, 
       Dongning Liu,
       Siqin Zhang,
       Yu Zhu,
       Luyao Teng,
       Shaohua Teng.

       https://www.sciencedirect.com/science/article/pii/S0304397516000037

       Programmers: Tom Shabalin and Dor Harizi.
       Date: 2024-05-16.
"""

import numpy as np
import logging

class ManyToManyAssignment:
       def __init__(self):
              self.taskRangeVector = None
              self.agentVector = None
              self.matrix = None
              self.colored_columns = []
              self.colored_rows = []
              self.zeroStars = []
              self.unavailable_list = []
              self.prime_Z = []
              self.unlcolored_zeros = []
              
       def many_to_many_assignment(self, taskRangeVector, agentVector, matrix)-> np.array:
              self.taskRangeVector = np.array(taskRangeVector)
              self.agentVector = np.array(agentVector)
              self.matrix = np.array(matrix)
              """
              Solving the Many to Many assignment problem by improving the Kuhn–Munkres algorithm with backtracking.

              Parameters
              ----------
              `ability_agent_vector`: Vector of the abilities of the agents
              `task_range_vector`: Vector of the task ranges
              `performance_matrix`: Matrix of the performance of the agents on the tasks

              Returns
              ----------
              Allocation Matrix of the agents to the tasks.

              Example 1: 
              >>> ability_agent_vector = np.array([1, 2, 1, 1])
              >>> task_range_vector = np.array([1, 1, 1, 1, 1])
              >>> performance_matrix = np.array([[49, 45, 39, 15, 16], [5, 30, 85, 22, 78], [61, 16, 71, 59, 20], [44, 79, 1, 48, 22]])
              >>> many_to_many_assignment(ability_agent_vector, task_range_vector, performance_matrix)
              array([[ 0,  0,  0,  0,  1],
                     [ 1,  0,  0,  1,  0],
                     [ 0,  1,  0,  0,  0],
                     [ 0,  0,  1,  0,  0]])
              
              Example 2:
              >>> ability_agent_vector = np.array([1, 1, 1])
              >>> task_range_vector = np.array([1, 1, 1])
              >>> performance_matrix = np.array([[40, 60, 15], [25, 30, 45], [55, 30, 25]])
              >>> preperation_stage(ability_agent_vector, task_range_vector, performance_matrix)
              array([[ 0,  0,  1],
                     [ 1,  0,  0],
                     [ 0,  1,  0]])

              Example 3: 
              >>> ability_agent_vector = np.array([1, 1, 1])
              >>> task_range_vector = np.array([1, 1, 1])
              >>> performance_matrix = np.array([[30, 25, 10], [15, 10, 20], [25, 20, 15]])
              >>> preperation_stage(ability_agent_vector, task_range_vector, performance_matrix)
              array([[ 0,  0,  1],
                     [ 0,  1,  0],
                     [ 1,  0,  0]])
              
              Example 4: 
              >>> ability_agent_vector = np.array([2, 2, 1, 3])
              >>> task_range_vector = np.array([1, 2, 3, 1, 1])
              >>> performance_matrix = np.array([[8, 6, 7, 9, 5], [6, 7, 8, 6, 7], [7, 8, 5, 6, 8], [7, 6, 9, 7, 5]])
              >>> preperation_stage(ability_agent_vector, task_range_vector, performance_matrix)
              ValueError: The Cordinality Constraint is not satisfied.

              Example 5: 
              >>> ability_agent_vector = np.array([1, 1, 1])
              >>> task_range_vector = np.array([1, 1, 1])
              >>> performance_matrix = np.array([[-30, 25, -1], [-7, 10, 2], [25, -4, -3]])
              >>> preperation_stage(ability_agent_vector, task_range_vector, performance_matrix)
              ValueError: The performance matrix has values less then 0.

              Example 6:
              >>> ability_agent_vector = np.array([1, 2, 1])
              >>> task_range_vector = np.array([1, 1, 1, 1])
              >>> performance_matrix = np.array([[3, 4, 5, 2], [5, 2, 3, 4], [1, 6, 2, 3]])
              >>> preperation_stage(ability_agent_vector, task_range_vector, performance_matrix)
              array([[ 0,  0,  0,  1],
                     [ 0,  1,  1,  0],
                     [ 0,  0,  1,  0]])
              """
              # Step 1: Preperation Stage
              try:
                     self.preperation_stage()
              except ValueError as e:
                     print(e)
                     return
              print(f'The matrix after the preperation stage:\n {self.matrix}')
              # Step 2: Find the minimum value in each row and subtract it from each element in the row.
                     #  Find the minimum value in each column and subtract it from each element in the column.
              self.find_min_value_in_row_and_subtruct()
              self.find_min_value_in_column_and_subtruct()
              print(f'The matrix after the first step:\n {self.matrix}')

              # Step 3: Find the first zero that is not 0*, mark it as 0* and mark the other 0 in the row and column as unavailable (if they exist).
              self.find_zero_star()
              print(f'0* values: {self.zeroStars}')
              print(f'Unavailable values: {self.unavailable_list}')
              is_all_colored = self.check_colored_columns()
              if is_all_colored:
                     print(f'Finished, going to step 7')
                     return
              print(f'Colored Columns: {self.colored_columns}')
              print(f'Is all colored: {is_all_colored}')

              # Step 4
              self.find_uncolored_zeros()
              while len(self.unlcolored_zeros) > 0:
                     print(f'Uncolored zeros: {self.unlcolored_zeros}')
                     break
              print(f'No more uncolored zeros')

       
       def duplicate_row(self, row_index: tuple):
              """
              Duplicate a row in a matrix.

              Parameters
              ----------
              `matrix`: The matrix to duplicate the row.
              `row_index (i, j)`: i - the row to duplicate, j - the number of times to duplicate.

              Returns
              ----------
              The matrix with the duplicated row.

              Example 1:
              >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
              >>> duplicate_row(matrix, (1, 1))
              array([[ 1,  2,  3],
                     [ 4,  5,  6],
                     [ 7,  8,  9],
                     [ 4,  5,  6]])

              Exampe 2:
              >>> matrix = np.array([[49, 45, 39, 15, 16], [5, 30, 85, 22, 78], [61, 16, 71, 59, 20], [44, 79, 1, 48, 22]])
              >>> duplicate_row(matrix, (1, 1))
              array([[ 49,  45,  39,  15,  16],
                     [ 5,  30,  85,  22,  78],
                     [ 61,  16,  71,  59,  20],
                     [ 44,  79,  1,  48,  22],
                     [ 5,  30,  85,  22,  78]])
              """
              row, num_duplications = row_index
              row_to_duplicate = self.matrix[row]
              duplicated_rows = np.tile(row_to_duplicate, (num_duplications - 1, 1))
              self.matrix = np.vstack((self.matrix, duplicated_rows))
       
       def create_zeros_columns(self, columns_amount: int):
              """
              Duplicate a column filled with zeros in a matrix.

              Parameters
              ----------
              `matrix`: The matrix to duplicate the column.
              `columns_amount`: The number of columns to duplicate.

              Returns
              ----------
              The matrix with the duplicated columns filled with zeros.

              Example 1:
              >>> matrix = np.array([[8, 6, 7, 9, 5], [6, 7, 8, 6, 7], [7, 8, 5, 6, 8], [7, 6, 9, 7, 5]])
              >>> create_zeros_columns(matrix, 2)
              array([[ 8,  6,  7,  9,  5,  0,  0],
                     [ 6,  7,  8,  6,  7,  0,  0],
                     [ 7,  8,  5,  6,  8,  0,  0],
                     [ 7,  6,  9,  7,  5,  0,  0]])
              
              Example 2:
              >>> matrix = np.array([[49, 45, 39, 15, 16], [5, 30, 85, 22, 78], [61, 16, 71, 59, 20], [44, 79, 1, 48, 22]])
              >>> create_zeros_columns(matrix, 1)
              array([[ 49,  45,  39,  15,  16,  0],
                     [ 5,  30,  85,  22,  78,  0],
                     [ 61,  16,  71,  59,  20,  0],
                     [ 44,  79,  1,  48,  22,  0]])
              """
              rows, _ = self.matrix.shape
              zeros = np.zeros((rows, columns_amount))
              self.matrix = np.hstack((self.matrix, zeros))

       def preperation_stage(self):
              """
              Preperation stage of the Matrix, and Cardinality Constraint detection.

              Parameters
              ----------
              `ability_agent_vector`: Vector of the abilities of the agents
              `task_range_vector`: Vector of the task ranges
              `performance_matrix`: Matrix of the performance of the agents on the tasks

              Returns
              ----------
              Matrix of the preperation stage

              Example 1: 
              >>> ability_agent_vector = np.array([1, 2, 1, 1])
              >>> task_range_vector = np.array([1, 1, 1, 1, 1])
              >>> performance_matrix = np.array([[49, 45, 39, 15, 16], [5, 30, 85, 22, 78], [61, 16, 71, 59, 20], [44, 79, 1, 48, 22]])
              >>> preperation_stage(ability_agent_vector, task_range_vector, performance_matrix)
              array([[ 49,  45,  39,  15,  16],
                     [ 5,  30,  85,  22,  78],
                     [ 61,  16,  71,  59,  20],
                     [ 44,  79,  1,  48,  22],
                     [ 5,  30,  85,  22,  78]])
              
              Example 2: 
              >>> ability_agent_vector = np.array([2, 2, 1, 3])
              >>> task_range_vector = np.array([1, 2, 1, 1, 1])
              >>> performance_matrix = np.array([[8, 6, 7, 9, 5], [6, 7, 8, 6, 7], [7, 8, 5, 6, 8], [7, 6, 9, 7, 5]])
              >>> preperation_stage(ability_agent_vector, task_range_vector, performance_matrix)
              array([[ 8,  6,  7,  9,  5,  0,  0,  0],
                     [ 6,  7,  8,  6,  7,  0,  0,  0],
                     [ 7,  8,  5,  6,  8,  0,  0,  0],
                     [ 7,  6,  9,  7,  5,  0,  0,  0],
                     [ 8,  6,  7,  9,  5,  0,  0,  0],
                     [ 7,  8,  5,  6,  8,  0,  0,  0],
                     [ 6,  7,  8,  6,  7,  0,  0,  0],
                     [ 7,  6,  9,  7,  5,  0,  0,  0],
                     [ 7,  6,  9,  7,  5,  0,  0,  0]])
              
              Example 3 (Failure): 
              >>> ability_agent_vector = np.array([2, 2, 1, 3])
              >>> task_range_vector = np.array([1, 2, 3, 1, 1])
              >>> performance_matrix = np.array([[8, 6, 7, 9, 5], [6, 7, 8, 6, 7], [7, 8, 5, 6, 8], [7, 6, 9, 7, 5]])
              >>> preperation_stage(ability_agent_vector, task_range_vector, performance_matrix)
              ValueError: The Cordinality Constraint is not satisfied.
              """
              agent_sum = sum(self.agentVector)
              task_sum = sum(self.taskRangeVector)

              for i in range(len(self.agentVector)):
                     if self.agentVector[i] > 1:
                            self.duplicate_row((i, self.agentVector[i]))
              if agent_sum < task_sum:
                     raise ValueError("The Cordinality Constraint is not satisfied.")
              print(f'\nThe Cordinality Constraint is satisfied with the values: {agent_sum} > {task_sum}\n')
              
              rows, columns = self.matrix.shape
              if rows > columns:
                     self.create_zeros_columns(rows - columns)

       def find_min_value_in_row_and_subtruct(self):
              """
              Finds the minimum value in each row and subtracts it from each element in the row.

              Parameters
              ----------
              `matrix`: The matrix to find the minimum value in each row and subtract it.

              Returns
              ----------
              The matrix with the minimum value in each row subtracted from each element in the row.

              Example 1:
              >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
              >>> find_min_value_in_row_and_subtruct(matrix)
              array([[ 0,  1,  2],
                     [ 0,  1,  2],
                     [ 0,  1,  2]])
              
              Example 2:
              >>> matrix = np.array([[49, 45, 39, 15, 16], [5, 30, 85, 22, 78], [61, 16, 71, 59, 20], [44, 79, 1, 48, 22]])
              >>> find_min_value_in_row_and_subtruct(matrix)
              array([[ 30,  26,  20,  0,  1],
                     [ 0,  25,  80,  0,  56],
                     [ 41,  0,  55,  43,  4],
                     [ 43,  78,  0,  47,  21]])
              """
              for i in range(self.matrix.shape[0]):
                     min_value = np.min(self.matrix[i])
                     self.matrix[i] -= min_value

       def find_min_value_in_column_and_subtruct(self):
              """
              Finds the minimum value in each column and subtracts it from each element in the column.

              Parameters
              ----------
              `matrix`: The matrix to find the minimum value in each column and subtract it.

              Returns
              ----------
              The matrix with the minimum value in each column subtracted from each element in the column.

              Example 1:
              >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
              >>> find_min_value_in_column_and_subtruct(matrix)
              array([[ 0,  1,  2],
                     [ 3,  4,  5],
                     [ 6,  7,  8]])
              
              Example 2:
              >>> matrix = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
              >>> find_min_value_in_column_and_subtruct(matrix)
              array([[ 0,  0,  0],
                     [ 0,  0,  0],
                     [ 0,  0,  0]])
              """
              for i in range(self.matrix.shape[1]):
                     min_value = np.min(self.matrix[:, i])
                     self.matrix[:, i] -= min_value

       def find_zero_star(self):
              """
              Traverse the matrix and find the first zero that is not 0*.
              Mark him as 0* and mark the other 0 in the row and column as unavailable (if they exist).

              Parameters
              ----------
              `matrix`: The matrix to find the first zero that is not 0*.
              `unavailable_dict`: The dictionary of the unavailable zeros.

              Returns
              ----------
              The matrix with after marking 0* and the unavailable zeros.

              Example 1:
              >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
              >>> unavailable_dict = {}
              >>> find_zero_star(matrix, unavailable_dict)
              (array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), {})

              Example 2:
              >>> matrix = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
              >>> unavailable_dict = {}
              >>> find_zero_star(matrix, unavailable_dict)
              (array([[0*, 1, 2], [0, 1, 2], [0, 1, 2]]), {0: (1, 0), 1: (2, 0)})
              """
              for i in range(self.matrix.shape[0]):
                     for j in range(self.matrix.shape[1]):
                            if self.matrix[i, j] == 0 and (i, j) not in self.zeroStars and (i, j) not in self.unavailable_list:
                                   self.zeroStars.append((i, j))
                                   self.colored_columns.append(j)
                                   self.mark_unavailable_zeros(i, j)
                                   break

       def color_row(self):
              """
              Color the rows of the matrix based on the condition:
              If there is 0* in the row of 0 in Prime_z then color the row and remove the color from the column of the 0*.

              Parameters
              ----------
              `matrix`: The matrix to color the rows.
              `prime_z`: Dictionary containing 0 values and their indexes.
              `colored_columns`: Vector containing True / False values at each index, which is colored / not colored for each column.

              Returns
              ----------
              Tuple of Vectors containing True / False values at each index, which is colored / not colored.
              First vector is the rows which is colored, and the second vector is the columns from which the color is removed.

              Example 1:
              >>> matrix = np.array([[0, 0*, 3], [4, 5, 6], [7, 8, 9]])
              >>> prime_z = {0: (0, 0)}
              >>> colored_columns = np.array([False, True, False])
              >>> color_row(matrix, prime_z, colored_columns)
              (array([ True, False, False]), array([False, False, False]))
              """
              pass

       def mark_unavailable_zeros(self, row: int, column: int):
              """
              Find a zero in the matrix that is not 0* and mark it as unavailable.

              Parameters
              ----------
              `matrix`: The matrix to find the zero that is not 0* and mark it as unavailable.
              `unavailable_dict`: The dictionary of the unavailable zeros.

              Returns
              ----------
              Dictionary with the unavailable zeros marked.

              Example 1:
              >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
              >>> unavailable_dict = {}
              >>> mark_unavailable_zeros(matrix)
              {}

              Example 2:
              >>> matrix = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
              >>> unavailable_dict = {}
              >>> mark_unavailable_zeros(matrix)
              {0: (1, 0), 1: (2, 0)}
              """
              for i in range(self.matrix.shape[0]):
                     if self.matrix[i, column] == 0 and (i, column) not in self.zeroStars:
                            self.unavailable_list.append((i, column))
              for j in range(self.matrix.shape[1]):
                     if self.matrix[row, j] == 0 and (row, j) not in self.zeroStars:
                            self.unavailable_list.append((row, j))

       def save_smallest_value(self):
              """
              Save the smallest value in the matrix (without unavailable values) such that his row or column is not colored.

              Parameters
              ----------
              `matrix`: The matrix to save the smallest value.
              `colored_rows`: Vector containing True / False values at each index, which is colored / not colored (ROWS)
              `colored_columns`: Vector containing True / False values at each index, which is colored / not colored (COLUMNS)
              `unavailable_dict`: The dictionary of the unavailable zeros.

              Returns
              ----------
              The smallest value in the matrix.

              Example 1:
              >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
              >>> colored_rows = np.array([False, False, False])
              >>> colored_columns = np.array([False, False, False])
              >>> unavailable_dict = {}
              >>> save_smallest_value(matrix, colored_rows, colored_columns)
              1

              Example 2:
              >>> matrix = np.array([[0*, 1, 4], [2, 0*, 2], [0, 1, 3]])
              >>> colored_rows = np.array([True, False, False])
              >>> colored_columns = np.array([False, True, False])
              >>> unavailable_dict = {0: (2, 0)}
              >>> save_smallest_value(matrix, colored_rows, colored_columns, unavailable_dict)
              2
              """
              pass

       def add_h_to_colored_row_elements(self):
              """
              Add the value h to each element in the row that is colored.

              Parameters
              ----------
              `matrix`: The matrix to add the value h to each element in the row that is colored.
              `colored_row_vector`: Vector containing True / False values at each index, which is colored / not colored.
              `h`: The value to add to each element in the row that is colored.

              Returns
              ----------
              The matrix with the value h added to each element in the row that is colored.

              Example 1:
              >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
              >>> colored_row_vector = np.array([True, False, False])
              >>> h = 2
              >>> add_h_to_colored_row_elements(matrix, colored_row_vector, h)
              array([[3, 4, 5],
                     [4, 5, 6],
                     [7, 8, 9]])
              """
              pass

       def substract_h_from_uncolored_columns(self):
              """
              Substract the value h from each element in the column that is not colored.

              Parameters
              ----------
              `matrix`: The matrix to substract the value h from each element in the column that is not colored.
              `colored_columns_vector`: Vector containing True / False values at each index, which is colored / not colored.
              `h`: The value to substract from each element in the column that is not colored.

              Returns
              ----------
              The matrix with the value h substracted from each element in the column that is not colored.

              Example 1:
              >>> matrix = np.array([[7, 4, 10], [4, 5, 6], [7, 8, 9]])
              >>> colored_columns_vector = np.array([False, True, False])
              >>> h = 2
              >>> substract_h_from_uncolored_columns(matrix, colored_columns_vector, h)
              array([[5,  4,  8],
                     [2,  5,  4],
                     [5,  8,  7]])
              """
              pass

       def check_colored_columns(self) -> bool:
              """
              Check if all the columns are colored.

              Parameters
              ----------
              `colored_columns`: Vector containing True / False values at each index, which is colored / not colored.

              Returns
              ----------
              True if all the columns are colored, otherwise False.

              Example 1:
              >>> colored_columns = np.array([True, True, True])
              >>> check_colored_columns(colored_columns)
              True

              Example 2:
              >>> colored_columns = np.array([True, False, True])
              >>> check_colored_columns(colored_columns)
              False
              """
              for i in range(self.matrix.shape[1]):
                     if not self.colored_columns[i]:
                            return False
              return True

       def find_uncolored_zeros(self):
              for i in range(self.matrix.shape[0]):
                     for j in range(self.matrix.shape[1]):
                            if self.matrix[i, j] == 0 and (i, j) not in self.zeroStars and (i, j) not in self.unavailable_list:
                                   self.unlcolored_zeros.append((i, j))
              

if __name__ == "__main__":
       """
       Example 1
       """
       ability_agent_vector = np.array([1, 2, 1, 1])
       task_range_vector = np.array([1, 1, 1, 1, 1])
       performance_matrix = np.array([[49, 45, 39, 15, 16], [5, 30, 85, 22, 78], [61, 16, 71, 59, 20], [44, 79, 1, 48, 22]])
       object = ManyToManyAssignment()
       object.many_to_many_assignment(task_range_vector, ability_agent_vector, performance_matrix)

       """
       Example 2
       """
       # ability_agent_vector = np.array([2, 2, 1, 3])
       # task_range_vector = np.array([1, 2, 1, 1, 1])
       # performance_matrix = np.array([[8, 6, 7, 9, 5], [6, 7, 8, 6, 7], [7, 8, 5, 6, 8], [7, 6, 9, 7, 5]])
       # object = ManyToManyAssignment()
       # object.many_to_many_assignment(task_range_vector, ability_agent_vector, performance_matrix)