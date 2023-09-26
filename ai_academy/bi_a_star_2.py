"""
A* algorithm
Author: Weicent
randomly generate obstacles, start and goal point
searching path from start and end simultaneously
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import yaml
from environment import get_main_file, get_obstacles
from spline3D import generate_curve
from bspline import B_spline


class Node:
    """node with properties of g, h, coordinate and parent node"""

    def __init__(self, G=0, H=0, coordinate=None, parent=None):
        self.G = G
        self.H = H
        self.F = G + H
        self.parent = parent
        self.coordinate = coordinate

    def reset_f(self):
        self.F = self.G + self.H


def min_max_scaler(data, input_range=(0, 20), output_range=(0, 1)):
    """
    Applies Min-Max scaling to the input data.
    
    Parameters:
        data (list or numpy.ndarray): The data to be scaled.
        feature_range (tuple): The desired range of the scaled data. Default is (0, 1).
        
    Returns:
        scaled_data (numpy.ndarray): The scaled data.
    """
    data = np.array(data)
    min_val, max_val = input_range
    range_min, range_max = output_range
    
    scaled_data = (data - min_val) / (max_val - min_val) * (range_max - range_min) + range_min
    
    return scaled_data


def pen_angle(data, dimension_3):
  linear_x = np.linspace(data[0][0], data[-1][0], len(data))
  linear_y = np.linspace(data[0][1], data[-1][1], len(data))
  linear_z = np.linspace(data[0][2], data[-1][2], len(data))

  cost = []
  for i in range(len(data)):
    diff_x = abs(data[i][0] - linear_x[i]) ** 2
    diff_y = abs(data[i][1] - linear_y[i]) ** 2
    if dimension_3:
        diff_z = abs(data[i][2] - linear_z[i]) ** 2
        cost.append(diff_x + diff_y + diff_z) 
    else:
        cost.append(diff_x + diff_y) 

  cost = min_max_scaler(cost, input_range=(min(cost), max(cost)), output_range=(0, 1/6))

  result = sum(cost)

  return result


def hcost(node_coordinate, goal, obstacles, dimension_3):
    dx = abs(node_coordinate[0] - goal[0])
    dy = abs(node_coordinate[1] - goal[1])
    if dimension_3:
        dz = abs(node_coordinate[2] - goal[2])
        hcost = dx + dy + dz
    else:
        hcost = dx + dy

    safe_distance = 2
    for obstacle in obstacles:
        if dimension_3:
            distance_to_obstacle = math.sqrt((node_coordinate[0] - obstacle[0])**2 +
                                             (node_coordinate[1] - obstacle[1])**2 +
                                             (node_coordinate[2] - obstacle[2])**2)
        else:
            distance_to_obstacle = math.sqrt((node_coordinate[0] - obstacle[0])**2 +
                                             (node_coordinate[1] - obstacle[1])**2)
        
        if distance_to_obstacle < safe_distance:
            hcost += hcost * min_max_scaler([safe_distance - distance_to_obstacle], input_range=(0, safe_distance))[0]

    return hcost


def gcost(fixed_node, update_node_coordinate, dimension_3):
    dx = abs(fixed_node.coordinate[0] - update_node_coordinate[0])
    dy = abs(fixed_node.coordinate[1] - update_node_coordinate[1])
    if dimension_3:
        dz = abs(fixed_node.coordinate[2] - update_node_coordinate[2])
        gc = math.hypot(dx, dy, dz)  # gc = move from fixed_node to update_node
    else:
        gc = math.hypot(dx, dy)

    try:
        # just enter if have at least 5 parents
        prev_node_1 = fixed_node.parent
        prev_node_2 = prev_node_1.parent
        prev_node_3 = prev_node_2.parent
        prev_node_4 = prev_node_3.parent
        prev_node_5 = prev_node_4.parent
        if dimension_3:
            recent_trajectory = [
                [prev_node_5.coordinate[0], prev_node_5.coordinate[1], prev_node_5.coordinate[2]], 
                [prev_node_4.coordinate[0], prev_node_4.coordinate[1], prev_node_5.coordinate[2]], 
                [prev_node_3.coordinate[0], prev_node_3.coordinate[1], prev_node_5.coordinate[2]], 
                [prev_node_2.coordinate[0], prev_node_2.coordinate[1], prev_node_5.coordinate[2]], 
                [prev_node_1.coordinate[0], prev_node_1.coordinate[1], prev_node_5.coordinate[2]], 
                [fixed_node.coordinate[0], fixed_node.coordinate[1], prev_node_5.coordinate[2]], 
            ]
        else:
            recent_trajectory = [
                [prev_node_5.coordinate[0], prev_node_5.coordinate[1]], 
                [prev_node_4.coordinate[0], prev_node_4.coordinate[1]], 
                [prev_node_3.coordinate[0], prev_node_3.coordinate[1]], 
                [prev_node_2.coordinate[0], prev_node_2.coordinate[1]], 
                [prev_node_1.coordinate[0], prev_node_1.coordinate[1]], 
                [fixed_node.coordinate[0], fixed_node.coordinate[1]], 
            ]
        gc += gc * pen_angle(recent_trajectory, dimension_3)
    except:
        pass


    gcost = fixed_node.G + gc  # gcost = move from start point to update_node
    return gcost


def find_neighbor(node, ob, closed, bottom_vertex, top_vertex, dimension_3):
    # generate neighbors in certain condition
    ob_list = ob.tolist()
    neighbor: list = []
    if dimension_3:
        for x in range(node.coordinate[0] - 1, node.coordinate[0] + 2):
            for y in range(node.coordinate[1] - 1, node.coordinate[1] + 2):
                for z in range(node.coordinate[2] - 1, node.coordinate[2] + 2):
                    x_boundaries_ok = top_vertex[0] > x > bottom_vertex[0]
                    y_boundaries_ok = top_vertex[1] > y > bottom_vertex[1]
                    z_boundaries_ok = top_vertex[2] > z > bottom_vertex[2]
                    if [x, y, z] not in ob_list and x_boundaries_ok and y_boundaries_ok and z_boundaries_ok:
                        # find all possible neighbor nodes
                        neighbor.append([x, y, z])
    else:
        for x in range(node.coordinate[0] - 1, node.coordinate[0] + 2):
            for y in range(node.coordinate[1] - 1, node.coordinate[1] + 2):
                if [x, y] not in ob_list:
                    # find all possible neighbor nodes
                    neighbor.append([x, y])
    # remove node violate the motion rule
    # 1. remove node.coordinate itself
    neighbor.remove(node.coordinate)
    # 2. remove neighbor nodes who cross through two diagonal
    # positioned obstacles since there is no enough space for
    # robot to go through two diagonal positioned obstacles

    if dimension_3:
        # top bottom left right neighbors of node
        top_nei = [node.coordinate[0], node.coordinate[1] + 1, node.coordinate[2]]
        bottom_nei = [node.coordinate[0], node.coordinate[1] - 1, node.coordinate[2]]
        left_nei = [node.coordinate[0] - 1, node.coordinate[1], node.coordinate[2]]
        right_nei = [node.coordinate[0] + 1, node.coordinate[1], node.coordinate[2]]
        # neighbors in four vertex
        lt_nei = [node.coordinate[0] - 1, node.coordinate[1] + 1, node.coordinate[2]]
        rt_nei = [node.coordinate[0] + 1, node.coordinate[1] + 1, node.coordinate[2]]
        lb_nei = [node.coordinate[0] - 1, node.coordinate[1] - 1, node.coordinate[2]]
        rb_nei = [node.coordinate[0] + 1, node.coordinate[1] - 1, node.coordinate[2]]

        # Up
        top_nei_up = [node.coordinate[0], node.coordinate[1] + 1, node.coordinate[2] + 1]
        bottom_nei_up = [node.coordinate[0], node.coordinate[1] - 1, node.coordinate[2] + 1]
        left_nei_up = [node.coordinate[0] - 1, node.coordinate[1], node.coordinate[2] + 1]
        right_nei_up = [node.coordinate[0] + 1, node.coordinate[1], node.coordinate[2] + 1]
        lt_nei_up = [node.coordinate[0] - 1, node.coordinate[1] + 1, node.coordinate[2] + 1]
        rt_nei_up = [node.coordinate[0] + 1, node.coordinate[1] + 1, node.coordinate[2] + 1]
        lb_nei_up = [node.coordinate[0] - 1, node.coordinate[1] - 1, node.coordinate[2] + 1]
        rb_nei_up = [node.coordinate[0] + 1, node.coordinate[1] - 1, node.coordinate[2] + 1]

        # Down
        top_nei_down = [node.coordinate[0], node.coordinate[1] + 1, node.coordinate[2] - 1]
        bottom_nei_down = [node.coordinate[0], node.coordinate[1] - 1, node.coordinate[2] - 1]
        left_nei_down = [node.coordinate[0] - 1, node.coordinate[1], node.coordinate[2] - 1]
        right_nei_down = [node.coordinate[0] + 1, node.coordinate[1], node.coordinate[2] - 1]
        lt_nei_down = [node.coordinate[0] - 1, node.coordinate[1] + 1, node.coordinate[2] - 1]
        rt_nei_down = [node.coordinate[0] + 1, node.coordinate[1] + 1, node.coordinate[2] - 1]
        lb_nei_down = [node.coordinate[0] - 1, node.coordinate[1] - 1, node.coordinate[2] - 1]
        rb_nei_down = [node.coordinate[0] + 1, node.coordinate[1] - 1, node.coordinate[2] - 1]
    else:
        # top bottom left right neighbors of node
        top_nei = [node.coordinate[0], node.coordinate[1] + 1]
        bottom_nei = [node.coordinate[0], node.coordinate[1] - 1]
        left_nei = [node.coordinate[0] - 1, node.coordinate[1]]
        right_nei = [node.coordinate[0] + 1, node.coordinate[1]]
        # neighbors in four vertex
        lt_nei = [node.coordinate[0] - 1, node.coordinate[1] + 1]
        rt_nei = [node.coordinate[0] + 1, node.coordinate[1] + 1]
        lb_nei = [node.coordinate[0] - 1, node.coordinate[1] - 1]
        rb_nei = [node.coordinate[0] + 1, node.coordinate[1] - 1]

    # remove the unnecessary neighbors
    if top_nei and left_nei in ob_list and lt_nei in neighbor:
        neighbor.remove(lt_nei)
    if top_nei and right_nei in ob_list and rt_nei in neighbor:
        neighbor.remove(rt_nei)
    if bottom_nei and left_nei in ob_list and lb_nei in neighbor:
        neighbor.remove(lb_nei)
    if bottom_nei and right_nei in ob_list and rb_nei in neighbor:
        neighbor.remove(rb_nei)

    if dimension_3:
        if top_nei_up and left_nei_up in ob_list and lt_nei_up in neighbor:
            neighbor.remove(lt_nei_up)
        if top_nei_up and right_nei_up in ob_list and rt_nei_up in neighbor:
            neighbor.remove(rt_nei_up)
        if bottom_nei_up and left_nei_up in ob_list and lb_nei_up in neighbor:
            neighbor.remove(lb_nei_up)
        if bottom_nei_up and right_nei_up in ob_list and rb_nei_up in neighbor:
            neighbor.remove(rb_nei_up)

        if top_nei_down and left_nei_down in ob_list and lt_nei_down in neighbor:
            neighbor.remove(lt_nei_down)
        if top_nei_down and right_nei_down in ob_list and rt_nei_down in neighbor:
            neighbor.remove(rt_nei_down)
        if bottom_nei_down and left_nei_down in ob_list and lb_nei_down in neighbor:
            neighbor.remove(lb_nei_down)
        if bottom_nei_down and right_nei_down in ob_list and rb_nei_down in neighbor:
            neighbor.remove(rb_nei_down)

    neighbor = [x for x in neighbor if x not in closed]
    return neighbor


def find_node_index(coordinate, node_list):
    # find node index in the node list via its coordinate
    ind = 0
    for node in node_list:
        if node.coordinate == coordinate:
            target_node = node
            ind = node_list.index(target_node)
            break
    return ind


def find_path(open_list, closed_list, goal, obstacle, bottom_vertex, top_vertex, dimension_3):
    # searching for the path, update open and closed list
    # obstacle = obstacle and boundary
    flag = len(open_list)
    for i in range(flag):
        node = open_list[0]
        open_coordinate_list = [node.coordinate for node in open_list]
        closed_coordinate_list = [node.coordinate for node in closed_list]
        temp = find_neighbor(node, obstacle, closed_coordinate_list, bottom_vertex, top_vertex, dimension_3)
        for element in temp:
            if element in closed_list:
                continue
            elif element in open_coordinate_list:
                # if node in open list, update g value
                ind = open_coordinate_list.index(element)
                new_g = gcost(node, element, dimension_3)
                if new_g <= open_list[ind].G:
                    open_list[ind].G = new_g
                    open_list[ind].reset_f()
                    open_list[ind].parent = node
            else:  # new coordinate, create corresponding node
                ele_node = Node(coordinate=element, parent=node,
                                G=gcost(node, element, dimension_3), H=hcost(element, goal, obstacle, dimension_3))
                open_list.append(ele_node)
        open_list.remove(node)
        closed_list.append(node)
        open_list.sort(key=lambda x: x.F)
    return open_list, closed_list


def node_to_coordinate(node_list):
    # convert node list into coordinate list and array
    coordinate_list = [node.coordinate for node in node_list]
    return coordinate_list


def check_node_coincide(close_ls1, closed_ls2):
    """
    :param close_ls1: node closed list for searching from start
    :param closed_ls2: node closed list for searching from end
    :return: intersect node list for above two
    """
    # check if node in close_ls1 intersect with node in closed_ls2
    cl1 = node_to_coordinate(close_ls1)
    cl2 = node_to_coordinate(closed_ls2)
    intersect_ls = [node for node in cl1 if node in cl2]
    return intersect_ls


def get_path(org_list, goal_list, coordinate):
    # get path from start to end
    path_org: list = []
    path_goal: list = []
    ind = find_node_index(coordinate, org_list)
    node = org_list[ind]
    while node != org_list[0]:
        path_org.append(node.coordinate)
        node = node.parent
    path_org.append(org_list[0].coordinate)
    ind = find_node_index(coordinate, goal_list)
    node = goal_list[ind]
    while node != goal_list[0]:
        path_goal.append(node.coordinate)
        node = node.parent
    path_goal.append(goal_list[0].coordinate)
    path_org.reverse()
    path = path_org + path_goal
    path = np.array(path)
    return path


def draw_control(org_closed, goal_closed, flag, start, end, bound, obstacle):
    """
    control the plot process, evaluate if the searching finished
    flag == 0 : draw the searching process and plot path
    flag == 1 or 2 : start or end is blocked, draw the border line
    """
    stop_loop = 0  # stop sign for the searching
    org_closed_ls = node_to_coordinate(org_closed)
    goal_closed_ls = node_to_coordinate(goal_closed)
    path = None
    if flag == 0:
        node_intersect = check_node_coincide(org_closed, goal_closed)
        if node_intersect:  # a path is find
            path = get_path(org_closed, goal_closed, node_intersect[0])
            stop_loop = 1
            print('Path found!')
    elif flag == 1:  # start point blocked first
        stop_loop = 1
        print('There is no path to the goal! Start point is blocked!')
    elif flag == 2:  # end point blocked first
        stop_loop = 1
        print('There is no path to the goal! End point is blocked!')
    return stop_loop, path


def one_step_per_time(path: list):
    new_path = []

    for node_in_path in path:
        if len(new_path) == 0:
            new_path.append(node_in_path.tolist())
            continue

        diff = node_in_path - new_path[-1]
        abs_diff = abs(diff)

        if sum(abs_diff) == 0:
            continue
        elif sum(abs_diff) == 1:
            new_path.append(node_in_path.tolist())
        else:
            non_zero_indexes = np.nonzero(diff)
            for index in non_zero_indexes[0]:
                new_node = new_path[-1].copy()
                new_node[index] += diff[index]
                new_path.append(new_node)

    return new_path


def transform_path_in_steps(path: list, dimension_3: bool):
    path_one_step = one_step_per_time(path)
    diff = np.diff(path_one_step, axis=0)
    result = []

    for d in diff:
        if d[1] > 0:
            result.append(0)  # front
        if d[1] < 0:
            result.append(1)  # back
        if d[0] > 0:
            result.append(2)  # right
        if d[0] < 0:
            result.append(3)  # left
        
        if dimension_3:
            if d[2] > 0:
                result.append(4)  # up
            if d[2] < 0:
                result.append(5)  # down

    return path_one_step, result


def searching_control(start, end, bound, obstacle, bottom_vertex, top_vertex, dimension_3):
    """manage the searching process, start searching from two side"""
    # initial origin node and end node
    origin = Node(coordinate=start, H=hcost(start, end, obstacle, dimension_3))
    goal = Node(coordinate=end, H=hcost(end, start, obstacle, dimension_3))
    # list for searching from origin to goal
    origin_open: list = [origin]
    origin_close: list = []
    # list for searching from goal to origin
    goal_open = [goal]
    goal_close: list = []
    # initial target
    target_goal = end
    # flag = 0 (not blocked) 1 (start point blocked) 2 (end point blocked)
    flag = 0  # init flag
    path = None
    while True:
        # searching from start to end
        origin_open, origin_close = \
            find_path(origin_open, origin_close, target_goal, bound, bottom_vertex, top_vertex, dimension_3)
        if not origin_open:  # no path condition
            flag = 1  # origin node is blocked
            draw_control(origin_close, goal_close, flag, start, end, bound,
                         obstacle)
            break
        # update target for searching from end to start
        target_origin = min(origin_open, key=lambda x: x.F).coordinate

        # searching from end to start
        goal_open, goal_close = \
            find_path(goal_open, goal_close, target_origin, bound, bottom_vertex, top_vertex, dimension_3)
        if not goal_open:  # no path condition
            flag = 2  # goal is blocked
            draw_control(origin_close, goal_close, flag, start, end, bound,
                         obstacle)
            break
        # update target for searching from start to end
        target_goal = min(goal_open, key=lambda x: x.F).coordinate

        # continue searching, draw the process
        stop_sign, path = draw_control(origin_close, goal_close, flag, start,
                                       end, bound, obstacle)
        if stop_sign:
            break
    return path


def create_boundaries(bottom_vertex, top_vertex, dimension_3):
    if dimension_3:
        # Calculate the ranges for each dimension
        ax = [bottom_vertex[0]] * (top_vertex[2] - bottom_vertex[2])
        ay = list(range(bottom_vertex[1], top_vertex[1]))
        az = [bottom_vertex[2]] * (top_vertex[1] - bottom_vertex[1])
        bx = list(range(bottom_vertex[0] + 1, top_vertex[0]))
        by = [bottom_vertex[1]] * (top_vertex[2] - bottom_vertex[2])
        bz = [top_vertex[2]] * (top_vertex[0] - bottom_vertex[0])
        cx = [top_vertex[0]] * (top_vertex[2] - bottom_vertex[2])
        cy = ay
        cz = az
        dx = [bottom_vertex[0]] * (top_vertex[1] - bottom_vertex[1])
        dy = [top_vertex[1]] * (top_vertex[0] - bottom_vertex[0])
        dz = bz

        # Calculate the x, y, and z coordinates in the specified order for the boundary
        x = ax + bx + cx + dx
        y = ay + by + cy + dy
        z = az + bz + cz + dz

        return x, y, z
    else:
        # below can be merged into a rectangle boundary
        ay = list(range(bottom_vertex[1], top_vertex[1]))
        ax = [bottom_vertex[0]] * len(ay)
        cy = ay
        cx = [top_vertex[0]] * len(cy)
        bx = list(range(bottom_vertex[0] + 1, top_vertex[0]))
        by = [bottom_vertex[1]] * len(bx)
        dx = [bottom_vertex[0]] + bx + [top_vertex[0]]
        dy = [top_vertex[1]] * len(dx)

        # x y coordinate in certain order for boundary
        x = ax + bx + cx + dx
        y = ay + by + cy + dy

        return x, y, []


def main_bi_astar(
        use_yaml: bool = True, 
        defined_yaml: bool = False, 
        options: list = ['rain_forest', 0], 
        show_animation: bool = False, 
        dimension_3: bool = False,
        user_start_node: list = [1,1,1],
        user_goal_node: list = [10,10,1]
    ):
    yaml_file = get_main_file()
    with open(f'{yaml_file}.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    start_node = data['start_node']
    goal_node = data['goal_node']
    env_type = data['env_type']
    env_number = data['env_number']
    higher_limit = data['higher_limit']
    lower_limit = data['lower_limit']

    if not use_yaml:
        start_node = {'x': user_start_node[0], 'y': user_start_node[1], 'z': user_start_node[2]}
        goal_node = {'x': user_goal_node[0], 'y': user_goal_node[1], 'z': user_goal_node[2]}
        obstacles, _, _, _ = get_obstacles(defined_yaml=False, options=[env_type, env_number])
    else:
        obstacles, _, _, _ = get_obstacles(defined_yaml=defined_yaml, options=options)

    top_vertex = [higher_limit['x'], higher_limit['y'], higher_limit['z']]  # top right vertex of boundary
    bottom_vertex = [lower_limit['x'], lower_limit['y'], lower_limit['z']]  # bottom left vertex of boundary

    if dimension_3:
        start = [start_node['x'], start_node['y'], start_node['z']]
        end = [goal_node['x'], goal_node['y'], start_node['z']]

        new_obstacles = [(_x, _y, _z) for _x, _y, _z in obstacles if _z == start[2]]


        obstacle = [coor for coor in new_obstacles if coor != start and coor != end]
        path = searching_control(start, end, np.array(obstacle), obstacle, bottom_vertex, top_vertex, dimension_3)
    else:
        x_bound, y_bound, _ = create_boundaries(bottom_vertex, top_vertex, dimension_3)

        start = [start_node['x'], start_node['y']]
        end = [goal_node['x'], goal_node['y']]

        new_obstacles = [(_x, _y) for _x, _y, _z in obstacles if _z == 2]

        obstacle = [coor for coor in new_obstacles if coor != start and coor != end]
        obs_array = np.array(obstacle)
        bound = np.vstack((x_bound, y_bound)).T
        bound_obs = np.vstack((bound, obs_array))
        path = searching_control(start, end, bound_obs, obstacle, bottom_vertex, top_vertex, dimension_3)
    
    path_one_step, steps = transform_path_in_steps(path, dimension_3)
    if not dimension_3:
        path_one_step = [(_x, _y, start_node['z']) for _x, _y in path_one_step]

    if show_animation and dimension_3:
        new_obstacles_np = np.array(new_obstacles)
        obs_x = new_obstacles_np[:, 0]
        obs_y = new_obstacles_np[:, 1]
        obs_z = new_obstacles_np[:, 2]

        traj_np = np.array(path_one_step)
        traj_x = traj_np[:, 0]
        traj_y = traj_np[:, 1]
        traj_z = traj_np[:, 2]

        traj_x_smooth, traj_y_smooth, traj_z_smooth = generate_curve(traj_x, traj_y, traj_z)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.clear()
        ax.scatter(obs_x, obs_y, obs_z, c='k', marker='s')
        ax.plot(traj_x, traj_y, traj_z)
        ax.plot(traj_x_smooth, traj_y_smooth, traj_z_smooth)
        ax.scatter([goal_node['x']], [goal_node['y']], [goal_node['z']], c='b', marker='*', label='Goal')
        ax.scatter([start_node['x']], [start_node['y']], [start_node['z']], c='b', marker='^', label='Origin')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xlim(lower_limit['x'], higher_limit['x'])
        ax.set_ylim(lower_limit['y'], higher_limit['y'])
        ax.set_zlim(lower_limit['z'], higher_limit['z'])
        ax.legend()
        plt.show()
    if show_animation and not dimension_3:
        new_obstacles_np = np.array(new_obstacles)
        obs_x = new_obstacles_np[:, 0]
        obs_y = new_obstacles_np[:, 1]

        traj_np = np.array(path)
        traj_x = traj_np[:, 0]
        traj_y = traj_np[:, 1]

        curv = B_spline(traj_x.tolist(), traj_y.tolist())
        traj_x_smooth, traj_y_smooth = curv.get_curv()

        plt.scatter(obs_x, obs_y, c='k', marker='s')
        plt.plot(traj_x, traj_y)
        plt.plot(traj_x_smooth, traj_y_smooth)
        plt.scatter([goal_node['x']], [goal_node['y']], c='b', marker='*', label='Goal')
        plt.scatter([start_node['x']], [start_node['y']], c='b', marker='^', label='Origin')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.xlim(lower_limit['x'], higher_limit['x'])
        plt.ylim(lower_limit['y'], higher_limit['y'])
        plt.legend()
        plt.show()
    
    return path_one_step, steps, obstacles


if __name__ == '__main__':
    path_one_step, steps, obstacles = main_bi_astar(defined_yaml=False, options=['rain_forest', 17])

    # print(path_one_step)
    # print('-----------')
    # print(steps)
