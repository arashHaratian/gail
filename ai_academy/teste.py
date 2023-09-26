# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import yaml
# from mpl_toolkits.mplot3d import Axes3D
# from environment import get_main_file, get_obstacles

# yaml_file = get_main_file()
# with open(f'{yaml_file}.yaml', 'r') as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)

# HIGHER_LIMIT = data['higher_limit']
# LOWER_LIMIT = data['lower_limit']
# GOAL_NODE = data['goal_node']

# # Parameters
# # TODO need to be optimized
# KP = 2.0  # attractive potential gain
# ETA = 15.0  # repulsive potential gain
# # AREA_WIDTH = 30.0  # potential area width [m]
# # the number of previous positions used to check oscillations
# # OSCILLATIONS_DETECTION_LENGTH = 3


# def calc_potential_field(gx, gy, gz, obstacles):
#     minx = LOWER_LIMIT['x']
#     miny = LOWER_LIMIT['y']
#     minz = LOWER_LIMIT['z']
#     maxx = HIGHER_LIMIT['x']
#     maxy = HIGHER_LIMIT['y']
#     maxz = HIGHER_LIMIT['z'] 
#     xw = int(round((maxx - minx)))
#     yw = int(round((maxy - miny)))
#     zw = int(round((maxz - minz)))

#     # calc each potential
#     pmap = np.zeros((xw, yw, zw)) # [[[0.0 for k in range(zw)] for j in range(yw)] for i in range(xw)]

#     for ix in range(xw):
#         x = ix + minx

#         for iy in range(yw):
#             y = iy + miny

#             for iz in range(zw):
#                 z = iz + minz

#                 ug = calc_attractive_potential(x, y, z, gx, gy, gz)
#                 uo = calc_repulsive_potential(x, y, z, obstacles)
#                 uf = ug + uo
#                 pmap[ix][iy][iz] = uf

#     return pmap, minx, miny, minz


# def calc_attractive_potential(x, y, z, gx, gy, gz):
#     return 0.5 * KP * math.hypot(x - gx, y - gy, z - gz)


# def calc_repulsive_potential(x, y, z, obstacles):
#     # search nearest obstacle
#     minid = -1
#     dmin = float("inf")
#     for i in range(len(obstacles)):
#         d = np.linalg.norm(np.array([x, y, z]) - np.array(obstacles[i]))
#         if dmin >= d:
#             dmin = d
#             minid = i

#     # calc repulsive potential
#     dq = np.linalg.norm(np.array([x, y, z]) - np.array(obstacles[minid]))
    
#     rr = 1
#     if dq <= rr:
#         if dq <= 0.1:
#             dq = 0.1

#         return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
#     else:
#         return 1.0


# def control_barrier_function(x, goal, obstacles, alpha):
#     """Compute the control input using a control barrier function."""
    
#     def distance(x, y):
#         """Compute the Euclidean distance between two points."""
#         return np.sqrt(np.sum((x - y)**2))
    
#     def gradient_distance(x, y):
#         """Compute the gradient of the distance function with respect to x."""
#         return (x - y) / distance(x, y)
    
#     # Compute the distance to the nearest obstacle
#     distances = [distance(x, o) for o in obstacles]
#     dist_to_obstacle = np.min(distances) # - robot_radius
    
#     # Compute the gradient of the control barrier function
#     grad_h = np.zeros_like(x)
#     if dist_to_obstacle <= 0:
#         # Robot is inside the safe set
#         return np.zeros_like(x)
#     else:
#         # Robot is outside the safe set
#         closest_obstacle = obstacles[np.argmin(distances)]
#         grad_d = gradient_distance(x, closest_obstacle)
#         grad_h = -grad_d / np.linalg.norm(x - closest_obstacle)
    
#     # Compute the control input
#     u_des = alpha * (goal - x) / np.linalg.norm(goal - x)
#     u = u_des + grad_h
    
#     return u


# def simulate_control_barrier_function(goal, obstacles, alpha):
#     """Simulate the control barrier function on a 2D grid."""
    
#     # Create a grid of values for each node
#     x = np.linspace(LOWER_LIMIT['x'], HIGHER_LIMIT['x'] - 1, HIGHER_LIMIT['x'])
#     y = np.linspace(LOWER_LIMIT['y'], HIGHER_LIMIT['y'] - 1, HIGHER_LIMIT['y'])
#     z = np.linspace(LOWER_LIMIT['z'], HIGHER_LIMIT['z'] - 1, HIGHER_LIMIT['z'])
#     X, Y, Z = np.meshgrid(x, y, z)
#     field_x = np.zeros_like(X)
#     field_y = np.zeros_like(Y)
#     field_z = np.zeros_like(Z)
    
#     # Compute the value of the control barrier function at each node
#     for i in range(HIGHER_LIMIT['x']):
#         for j in range(HIGHER_LIMIT['y']):
#             for k in range(HIGHER_LIMIT['z']):
#                 pos = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
#                 cbf = control_barrier_function(pos, goal, obstacles, alpha)
#                 field_x[i][j][k] = cbf[0]
#                 field_y[i][j][k] = cbf[1]
#                 field_z[i][j][k] = cbf[2]

#     return field_x, field_y, field_z


# def multiply_magnetic_fielf_per_cbf(matrix: np.array, field_x: np.array, field_y: np.array) -> np.array:
#     max_x = len(field_x)
#     max_y = len(field_y)
#     for i in range(max_x):
#         for j in range(max_y):
#             new_index_x = i - 1 if field_x[i][j] < 0 else i + 1
#             new_index_y = j - 1 if field_y[i][j] < 0 else j + 1

#             if max_x > new_index_x > 0 and max_y > new_index_y > 0:
#               if field_x[i][j] is np.nan or field_y[i][j] is np.nan:
#                 matrix[i][j] = 5
#               else:
#                 matrix[new_index_x][j] *= field_x[i][j] 
#                 matrix[i][new_index_y] *= field_y[i][j] 

#     return matrix


# def plot_heatmap(matrix: np.array, title_name: str = ''):
#     # Plot the heatmap
#     fig, ax = plt.subplots()
#     im = ax.imshow(matrix, cmap='viridis')

#     # Add a colorbar
#     cbar = ax.figure.colorbar(im, ax=ax)

#     # Set the ticks and tick labels
#     ax.set_xticks(np.arange(matrix.shape[1]))
#     ax.set_yticks(np.arange(matrix.shape[0]))
#     ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1))
#     ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1))

#     # Rotate the tick labels and set their alignment
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     # Loop over the data and create the annotations
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             text = ax.text(j, i, round(matrix[i, j], 2), ha="center", va="center", color="w")

#     ax.set_xlim(LOWER_LIMIT['x'] - 0.5, HIGHER_LIMIT['x'] - 0.5)
#     ax.set_ylim(LOWER_LIMIT['y'] - 0.5, HIGHER_LIMIT['y'] - 0.5)
#     ax.set_title(title_name)
#     plt.show()


# def main():
#     gx = GOAL_NODE['x']
#     gy = GOAL_NODE['y']
#     gz = GOAL_NODE['z']

#     obstacles, _, _, _ = get_obstacles()

#     pmap, _, _, _ = calc_potential_field(gx, gy, gz, obstacles)

#     matrix = np.array(pmap)
#     matrix = 1 / matrix # * 20

#     field_x, _, _ = simulate_control_barrier_function(np.array([gx, gy, gz]), obstacles, 1)

#     # matrix_ = multiply_magnetic_fielf_per_cbf(matrix, field_x, field_y)

#     # plot_heatmap(matrix)
#     # plot_heatmap(matrix * field_x)
#     # plot_heatmap(field_x, 'field_x')
#     # plot_heatmap(field_y, 'field_y')

#     rounded_matrix = np.round(matrix, decimals=2)
#     print(rounded_matrix.shape)
#     print(rounded_matrix)

#     matrix = matrix * field_x
#     height, width, depth = matrix.shape
#     x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth))
#     values = []
#     for _x, _y, _z in zip(x.ravel(), y.ravel(), z.ravel()):
#         values.append(matrix[_x, _y, _z])

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = ax.scatter(x, y, z, vmin=0.0, vmax=0.5, c=values, cmap='YlOrRd', linewidths=5)

#     obstacles = [[2, 2, 2], [3, 3, 2], [4, 4, 2], [5, 5, 2],
#                      [2, 2, 3], [3, 3, 3], [4, 4, 3], [5, 5, 3],
#                      [2, 1, 2], [3, 1, 2], [4, 1, 2], [5, 1, 2],
#                      [2, 1, 3], [3, 1, 3], [4, 1, 3], [5, 1, 3],
#                      [1, 4, 2], [1, 5, 2], [1, 4, 3], [1, 5, 3]
#                      ]
#     ox = [o[0] for o in obstacles]
#     oy = [o[1] for o in obstacles]
#     oz = [o[2] for o in obstacles]

#     ax.scatter([1], [1], [2], c='r', linewidths=10)
#     ax.scatter([6], [6], [4], c='r', linewidths=10)
#     ax.scatter(ox, oy, oz, c='k', linewidths=8)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Add colorbar
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Values')

#     plt.show()
    
#     return matrix * field_x
    

# if __name__ == '__main__':
#     main()




# import numpy as np
# x = [1,2,3,4,5]
# y = [1,2,3,4,5]
# z = [1,2,3,4,5]
# data = np.vstack((x, y, z))
# np.savetxt('arrays1.txt', data, fmt='%.2f')





import numpy as np
import matplotlib.pyplot as plt

# Leitura dos dados do arquivo
data = np.loadtxt('arrays.txt')

# Separação dos dados em x, y e z
x = data[0]
y = data[1]
z = data[2]

# Plotagem do gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

print(x)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()