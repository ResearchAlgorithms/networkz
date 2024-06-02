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

import math
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
              self.unavailable_zeros = []
              self.prime_z = []
              self.unlcolored_zeros = []
              self.Z_0 = []
              self.Z_1 = []
              self.Z_2 = []
              self.min_value = None
              
       def many_to_many_assignment(self, taskRangeVector, agentVector, matrix)-> np.array:
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
              self.taskRangeVector = np.array(taskRangeVector)
              self.agentVector = np.array(agentVector)
              self.matrix = np.array(matrix)
              # Step 1: Preperation Stage
              try:
                     self.preperation_stage()
              except ValueError as e:
                     logging.warning(e)
                     return
              logging.info(f'The matrix after Step 1:\n {self.matrix}')
              # Step 2: Find the minimum value in each row and subtract it from each element in the row.
                     #  Find the minimum value in each column and subtract it from each element in the column.
              self.find_min_value_in_row_and_subtruct()
              self.find_min_value_in_column_and_subtruct()
              logging.info(f'The matrix after the first step:\n {self.matrix}\n')
              self.find_zero_star()
              logging.info(f'The zero stars: {self.zeroStars}.\nStart of the main while loop.\n')
              # Main loop
              while True:
                     if self.check_colored_columns():
                            self.construct_final_solution()
                            return
                     # Inner loop
                     while not self.check_colored_columns():
                            self.find_uncolored_zeros()
                            for row, column in self.unlcolored_zeros:
                                   if (row, column) not in self.prime_z:
                                          self.prime_z.append((row, column))
                                          self.mark_unavailable_zeros(row, column)
                                          for zero_star in self.zeroStars:
                                                 if zero_star[0] == row and row not in self.colored_rows:
                                                        self.colored_rows.append(row)
                                                        if zero_star[1] in self.colored_columns:
                                                               self.colored_columns.remove(zero_star[1])
                                                        break
                                   else:
                                          # while column in [zero_star[1] for zero_star in self.zeroStars]:
                                          for i, j in self.prime_z:
                                                 if j in [zero_star[1] for zero_star in self.zeroStars]:
                                                        # Create Z_0: all 0 in prime_z and in uncolored_zeros
                                                        if (i, j) in self.unlcolored_zeros and (i, j) not in self.Z_0:
                                                               self.Z_0.append((i, j))
                                                        # Create Z_1: all 0 in zero_star group and in columns of Z_0
                                                        for k, m in self.zeroStars:
                                                               if m in [z_0[1] for z_0 in self.Z_0] and (k, m) not in self.Z_1:
                                                                      self.Z_1.append((k, m))
                                                        # Create Z_2: all 0 in prime_z and in rows of Z_1
                                                        for r, t in self.prime_z:
                                                               if r in [z_1[0] for z_1 in self.Z_1] and (r, t) not in self.Z_2:
                                                                      self.Z_2.append((r, t))
                                          # Unstar all 0* that is in Z_0, Z_1, Z_2 (if exist)
                                          temp_backtracking = []
                                          for row, column in self.zeroStars:
                                                 if (row, column) in self.Z_0 or (row, column) in self.Z_1 or (row, column) in self.Z_2:
                                                        temp_backtracking.append((row, column))
                                                        self.zeroStars.remove((row, column))
                                          # Star all 0 that in prime_Z and in Z_0, Z_1, Z_2
                                          for row, column in self.prime_z:
                                                 if (row, column) in self.Z_0 or (row, column) in self.Z_1 or (row, column) in self.Z_2:
                                                        self.zeroStars.append((row, column))

                                          # Clean up prime_z and clean up colored_rows
                                          self.prime_z = []
                                          self.colored_rows = []

                                          # Backtracking: Adjust unavailable elements to be available according to the erased stared zeros
                                          # According to the erased starred and primed zeros.
                                          for row, column in temp_backtracking:
                                                 for i, j in self.unavailable_zeros:
                                                        if i == row or j == column:
                                                               self.unavailable_zeros.remove((i, j))
                                          self.step_3_func()
                                   # End if loop
                            # End for non covered zero loop
                            self.save_smallest_value()
                            self.add_h_to_colored_row_elements(self.min_value)
                            self.substract_h_from_uncolored_columns(self.min_value)
                     # End of while inner loop
              # End of while main loop
              # End of the Algorithm



       
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
              """
              agent_sum = sum(self.agentVector)
              task_sum = sum(self.taskRangeVector)

              for i in range(len(self.agentVector)):
                     if self.agentVector[i] > 1:
                            self.duplicate_row((i, self.agentVector[i]))
              if agent_sum < task_sum:
                     warning_message = "The Cordinality Constraint is not satisfied."
                     logging.warning(warning_message)
                     raise ValueError(warning_message)
              logging.info(f'The Cordinality Constraint is satisfied with the values: {agent_sum} > {task_sum}')
              
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
              """
              for i in range(self.matrix.shape[0]):
                     for j in range(self.matrix.shape[1]):
                            if self.matrix[i, j] == 0 and (i, j) not in self.zeroStars and (i, j) not in self.unavailable_zeros:
                                   self.zeroStars.append((i, j))
                                   self.colored_columns.append(j)
                                   self.mark_unavailable_zeros(i, j)
                                   break

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
              for index_col in range(self.matrix.shape[1]):
                     if self.matrix[row, index_col] == 0 and (row, index_col) not in self.zeroStars and (row, index_col) not in self.unavailable_zeros and (row, index_col) not in self.prime_z:
                            self.unavailable_zeros.append((row, index_col))
              for index_row in range(self.matrix.shape[0]):
                     if self.matrix[index_row, column] == 0 and (index_row, column) not in self.zeroStars and (index_row, column) not in self.unavailable_zeros and (index_row, column) not in self.prime_z:
                            self.unavailable_zeros.append((index_row, column))

       def save_smallest_value(self) -> float | None:
              """
              Save the smallest value in the matrix (without unavailable values) such that his row or column is not colored.
              """
              self.min_value = math.inf
              for i in range(self.matrix.shape[0]):
                     for j in range(self.matrix.shape[1]):
                            if (i, j) not in self.unavailable_zeros and i not in self.colored_rows and j not in self.colored_columns and (i, j) not in self.zeroStars:
                                   if self.matrix[i,j] < self.min_value:
                                          self.min_value = self.matrix[i,j]

       def add_h_to_colored_row_elements(self, h: float):
              """
              Add the value h to each element in the row that is colored.
              """
              for row in range(self.matrix.shape[0]):
                     for column in range(self.matrix.shape[1]):
                            if row in self.colored_rows and (row, column) not in self.unavailable_zeros and (row, column) not in self.zeroStars:
                                   self.matrix[row, column] += h

       def substract_h_from_uncolored_columns(self, h: float) -> None:
              """
              Substract the value h from each element in the column that is not colored.
              """
              for row in range(self.matrix.shape[0]):
                     for column in range(self.matrix.shape[1]):
                            if column not in self.colored_columns and (row, column) not in self.unavailable_zeros and (row, column) not in self.zeroStars:
                                   self.matrix[row, column] -= h

       def check_colored_columns(self) -> bool:
              """
              Check if all the columns are colored.
              """
              if len(self.colored_columns) == len(self.matrix):
                     return True
              return False

       def find_uncolored_zeros(self):
              self.unlcolored_zeros = []
              for i in range(self.matrix.shape[0]):
                     for j in range(self.matrix.shape[1]):
                            if self.matrix[i, j] == 0 and (i, j) not in self.zeroStars and (i, j) not in self.unavailable_zeros:
                                   if i not in self.colored_rows and j not in self.colored_columns:
                                          self.unlcolored_zeros.append((i, j))

       def step_3_func(self):
              for _, column in self.zeroStars:
                     if column not in self.colored_columns:
                            self.colored_columns.append(column)
              if self.check_colored_columns():
                     self.construct_final_solution()

       def construct_final_solution(self):
              """
              Construct the final solution of the assignment as a matrix of 0 and 1 containing the starred zeros.
              """
              final_solution = np.zeros(self.matrix.shape, dtype=int)
              for i, j in self.zeroStars:
                     final_solution[i, j] = 1
                     logging.info(f'The agent: {i} is assigned to the task: {j}\n')
              logging.info(f'The final solution is:\n {final_solution}')
              return final_solution

if __name__ == "__main__":
       """
       Example 1
       """
       logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
       # ability_agent_vector = np.array([1, 2, 1, 1])
       # task_range_vector = np.array([1, 1, 1, 1, 1])
       # performance_matrix = np.array([[49, 45, 39, 15, 16], [5, 30, 85, 22, 78], [61, 16, 71, 59, 20], [44, 79, 1, 48, 22]])
       # object = ManyToManyAssignment()
       # object.many_to_many_assignment(task_range_vector, ability_agent_vector, performance_matrix)

       """
       Example 2
       """
       ability_agent_vector = np.array([2, 2, 1, 3])
       task_range_vector = np.array([1, 2, 1, 1, 1])
       performance_matrix = np.array([[8, 6, 7, 9, 5], [6, 7, 8, 6, 7], [7, 8, 5, 6, 8], [7, 6, 9, 7, 5]])
       object = ManyToManyAssignment()
       object.many_to_many_assignment(task_range_vector, ability_agent_vector, performance_matrix)