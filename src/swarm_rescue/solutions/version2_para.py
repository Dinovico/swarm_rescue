import math
from copy import deepcopy
from typing import Optional

import numpy as np
import heapq
import random

from queue import Queue

import threading
import multiprocessing

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None
    
    def __lt__(self, other):
        return self.f < other.f
    

class GridMap:
    def __init__(self, size, resolution):
        self.size = size  # size of the map (in pixels)
        self.resolution = resolution  # size of each cell (in pixels)
        self.rows = int(size[0] / resolution) + 2
        self.cols = int(size[1] / resolution) + 2
        self.map = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Ajout des bordures de la carte
        for i in range(self.rows):
            if i == 0 or i == self.rows-1:
                for j in range(self.cols):
                    self.map[i][j] = 1
            else:
                self.map[i][0] = 1
                self.map[i][self.cols-1] = 1
        
        print(self.rows, self.cols)
    

    def update_wall(self, x, y):
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return
        self.map[x][y] = 1

    def update_cell_danger(self, x, y, danger):
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return
        if self.map[x][y] in [1, -1]:
            return
        for i in range(max(0, x-5), min(x+6, self.rows)):
            for j in range(max(0, y-5), min(y+6, self.cols)):
                if i == x and j == y:
                    self.map[i][j] = danger
                else:
                    dist = max(abs(i-x), abs(j-y))
                    self.map[i][j] = max(min(self.map[x][y] + (danger * ((1 - abs(self.map[x][y])) / (dist + 1))), 1), -1)



    def random_cell_weighted_by_danger(self):
        # Création d'une liste contenant toutes les cellules de la grille avec leur poids respectif
        cells = []
        for i in range(self.rows):
            for j in range(self.cols):
                weight = 1 - abs(self.map[i][j])
                cells.append(((i, j), weight))
        
        # Tirage au sort d'une cellule pondérée par son poids
        total_weight = sum(weight for cell, weight in cells)
        rand = random.uniform(0, total_weight)
        for cell, weight in cells:
            if rand < weight:
                return cell
            rand -= weight
        
        # Si on arrive ici, c'est qu'il y a eu une erreur (par exemple, toutes les cellules ont un poids nul)
        raise Exception("Unable to select a random cell weighted by danger")
    
    def gps_to_grid_cell(self, gps_pos):
        if np.isnan(gps_pos[0]) or np.isnan(gps_pos[1]):
            # Si la position GPS est invalide, retourner une cellule hors de la grille
            return (-1, -1)
        else:
            x = gps_pos[0]
            y = gps_pos[1]
            row = int((x + self.size[0] / 2) / self.resolution)
            col = int((y + self.size[1] / 2) / self.resolution)
            return (row, col)
    
    def grid_cell_to_gps(self, cell_pos):
        """
        Convertit des indices de cellules dans la représentation sous forme de grille en coordonnées GPS.
        """
        row, col = cell_pos
        x = (row * self.resolution) - (self.size[0] / 2) + (self.resolution / 2)
        y = (col * self.resolution) - (self.size[1] / 2) + (self.resolution / 2)
        return (x, y)
    




def a_star(grid_map, start_x, start_y, end_x, end_y):
    # Création des noeuds de départ et d'arrivée
    start_node = Node(start_x, start_y)
    end_node = Node(end_x, end_y)
    
    # Initialisation de la liste ouverte et de l'ensemble fermé
    open_list = []
    heapq.heappush(open_list, start_node)
    closed_set = set()
    
    # Boucle principale de l'algorithme A*
    while open_list:
        # Récupération du noeud ayant le coût total (f) le plus faible dans la liste ouverte
        current_node = heapq.heappop(open_list)
        
        # Si on est arrivé à destination, on reconstruit le chemin en remontant les parents de chaque noeud
        if current_node.x == end_node.x and current_node.y == end_node.y:
            path = []
            while current_node.parent:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            path.append((start_x, start_y))
            return list(reversed(path))
        
        # Ajout du noeud courant à l'ensemble fermé
        closed_set.add((current_node.x, current_node.y))
        
        # Exploration des voisins du noeud courant
        neighbors = [(current_node.x-1, current_node.y), (current_node.x+1, current_node.y), (current_node.x, current_node.y-1), (current_node.x, current_node.y+1)]
        num_threads = multiprocessing.cpu_count()
        queue = Queue()
        for i in range(num_threads):
            t = threading.Thread(target=process_neighbors, args=(queue, grid_map, current_node, end_x, end_y, closed_set, neighbors[i::num_threads], open_list))
            t.start()
        queue.join()
    
    # Si on n'a pas trouvé de chemin, on renvoie None
    return None

def process_neighbors(queue, grid_map, current_node, end_x, end_y, closed_set, neighbors, open_list):
    for neighbor_x, neighbor_y in neighbors:
        # Vérification que le voisin est dans la grille
        if neighbor_x < 0 or neighbor_x >= grid_map.rows or neighbor_y < 0 or neighbor_y >= grid_map.cols:
            continue
        # Vérification que le voisin n'est pas un obstacle
        if grid_map.map[neighbor_x][neighbor_y] == 1:
            continue
        # Vérification que le voisin n'a pas déjà été exploré
        if (neighbor_x, neighbor_y) in closed_set:
            continue
        
        # Création d'un nouveau noeud pour le voisin
        neighbor_node = Node(neighbor_x, neighbor_y)
        # Calcul du coût de déplacement depuis le noeud courant
        neighbor_node.g = current_node.g + 1
        # Calcul de la valeur heuristique pour estimer le coût restant jusqu'à l'arrivée
        neighbor_node.h = 1 + min([grid_map.map[nx][ny] for nx, ny in [(neighbor_x-1, neighbor_y), (neighbor_x+1, neighbor_y), (neighbor_x, neighbor_y-1), (neighbor_x, neighbor_y+1)] if 0 <= nx < grid_map.rows and 0 <= ny < grid_map.cols])
        # Calcul du coût total (f)
        neighbor_node.f = neighbor_node.g + neighbor_node.h * (abs(end_x-neighbor_x) + abs(end_y-neighbor_y))
        # Mise à jour du parent du noeud voisin
        neighbor_node.parent = current_node
        
        # Si le voisin est déjà dans la liste ouverte, on met à jour ses coûts et son parent si nécessaire
        if neighbor_node in open_list:
            existing_node = open_list[open_list.index(neighbor_node)]
            if neighbor_node.g < existing_node.g:
                existing_node.g = neighbor_node.g
                existing_node.f = neighbor_node.f
                existing_node.parent = neighbor_node.parent
        # Sinon, on ajoute le voisin à la liste ouverte
        else:
            heapq.heappush(open_list, neighbor_node)
    queue.task_done()



class MyDroneV2(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=False,
                         **kwargs)
        self.grid = GridMap(size=misc_data.size_area, resolution=100)
        self.target_cell = None  # La cellule cible que le drone doit atteindre
        self.path_to_target = []  # Le chemin à suivre pour atteindre la cible

    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        return msg_data


    def control(self):
        """
        In this example, we only use the lidar sensor and the communication to move the drone
        The idea is to make the drones move like a school of fish.
        The lidar will help avoid running into walls.
        The communication will allow to know the position of the drones in the vicinity, to then correct its own
        position to stay at a certain distance and have the same orientation.
        """

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0}

        command_lidar, collision_lidar, walls = self.process_lidar_sensor(
            self.lidar())
        found, command_comm = self.process_communication_sensor()

        alpha = 0.4
        alpha_rot = 0.75

        gps_pos = self.measured_gps_position()


        grid_pos = self.grid.gps_to_grid_cell(gps_pos)
        print("grid pos: ", grid_pos, " gps pos: ", gps_pos)

        
        if collision_lidar:
            alpha_rot = 0.1

        # The final command  is a combination of 2 commands
        command["forward"] = \
            alpha * command_comm["forward"] \
            + (1 - alpha) * command_lidar["forward"]
        command["lateral"] = \
            alpha * command_comm["lateral"] \
            + (1 - alpha) * command_lidar["lateral"]
        command["rotation"] = \
            alpha_rot * command_comm["rotation"] \
            + (1 - alpha_rot) * command_lidar["rotation"]
        

        

        
        
        if ((np.isnan(gps_pos[0]) or np.isnan(gps_pos[1])) == False):
            # Mettre à jour le danger de la cellule actuelle
            self.grid.update_cell_danger(grid_pos[0], grid_pos[1], danger=-1)


            # Si le drone n'a pas encore de cible, il en choisit une au hasard
            while(self.target_cell is None or self.path_to_target is None):
                self.target_cell = self.grid.random_cell_weighted_by_danger()
                self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1])
                print(self.path_to_target)


            if (len(walls) > 0):
                all_walls_detected = True
                for i in range(len(walls)):
                    wall_cell = self.grid.gps_to_grid_cell(walls[i])
                    if (self.grid.map[wall_cell[0]][wall_cell[1]] != 1):
                        self.grid.update_wall(wall_cell[0], wall_cell[1])
                        all_walls_detected = False
                if not all_walls_detected:
                    while(True):
                        self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1])
                        if not (self.path_to_target is None):
                            break
                        self.target_cell = self.grid.random_cell_weighted_by_danger()
                
            # Si le drone est proche de sa cible, il en choisit une nouvelle au hasard
            distance_to_target = np.linalg.norm(np.array(gps_pos) - np.array(self.target_cell))
            if distance_to_target < 10:
                self.target_cell = self.grid.random_cell_weighted_by_danger()
                self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1])

            # Le drone suit le chemin vers sa cible
            if len(self.path_to_target) > 0:
                print("next cell: ", self.path_to_target[0], " in GPS: ", self.grid.grid_cell_to_gps(self.path_to_target[0]))
                target_direction = np.array(self.grid.grid_cell_to_gps(self.path_to_target[0])) - np.array(gps_pos)
                target_direction_norm = target_direction / np.linalg.norm(target_direction)
                print("target direction: ", target_direction, " target direction normalised: ", target_direction_norm)
                angle_to_target = np.arctan2(target_direction_norm[1], target_direction_norm[0])
                print("angle to target: ", angle_to_target, " current angle: ", self.measured_compass_angle())

                if (angle_to_target - self.measured_compass_angle()[0] < np.pi and angle_to_target - self.measured_compass_angle()[0] > 0):
                    command["rotation"] = 1
                else:
                    command["rotation"] = -1

                print("distance cible: ", np.linalg.norm(target_direction), " seuil: ", ( (self.grid.size[0] / self.grid.rows) / 2))
                #print("(0, 0) cell: ", self.grid.grid_cell_to_gps((0,0)))
                #print("(-350.0, -200.0) gps: ", self.grid.gps_to_grid_cell((-350.0, -200.0)))
                if np.linalg.norm(target_direction) < ( (self.grid.size[0] / self.grid.cols) / 2):
                    self.path_to_target.pop(0)
                    print(self.path_to_target)

        print(" ")
        return command

    def process_lidar_sensor(self, the_lidar_sensor):
        command = {"forward": 1.0,
                "lateral": 0.0,
                "rotation": 0.0}
        angular_vel_controller = 1.0

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return command, False

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution

        far_angle_raw = 0
        near_angle_raw = 0
        min_dist = 1000
        collision_points = []
        
        if size != 0:
            # far_angle_raw : angle with the longer distance
            far_angle_raw = ray_angles[np.argmax(values)]
            min_dist = min(values)
            # near_angle_raw : angle with the nearest distance
            near_angle_raw = ray_angles[np.argmin(values)]

            for i in range(size):
                if values[i] < 300:
                    x = values[i] * np.cos(ray_angles[i])
                    y = values[i] * np.sin(ray_angles[i])
                    collision_points.append((x, y))

        far_angle = far_angle_raw
        # If far_angle_raw is small then far_angle = 0
        if abs(far_angle) < 1 / 180 * np.pi:
            far_angle = 0.0

        near_angle = near_angle_raw
        far_angle = normalize_angle(far_angle)

        """
        # The drone will turn toward the zone with the more space ahead
        if size != 0:
            if far_angle > 0:
                command["rotation"] = angular_vel_controller
            elif far_angle == 0:
                command["rotation"] = 0
            else:
                command["rotation"] = -angular_vel_controller
        

        # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
        collision = False
        if size != 0 and min_dist < 50:
            collision = True
            if near_angle > 0:
                command["rotation"] = -angular_vel_controller
            else:
                command["rotation"] = angular_vel_controller
        """

        return command, False, collision_points


    def process_communication_sensor(self):
        found_drone = False
        command_comm = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0}

        if self.communicator:
            received_messages = self.communicator.received_messages
            nearest_drone_coordinate1 = (
                self.measured_gps_position(), self.measured_compass_angle())
            nearest_drone_coordinate2 = deepcopy(nearest_drone_coordinate1)
            (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
            (nearest_position2, nearest_angle2) = nearest_drone_coordinate2

            min_dist1 = 10000
            min_dist2 = 10000
            diff_angle = 0

            # Search the two nearest drones around
            for msg in received_messages:
                message = msg[1]
                coordinate = message[1]
                (other_position, other_angle) = coordinate

                dx = other_position[0] - self.measured_gps_position()[0]
                dy = other_position[1] - self.measured_gps_position()[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # if another drone is near
                if distance < min_dist1:
                    min_dist2 = min_dist1
                    min_dist1 = distance
                    nearest_drone_coordinate2 = nearest_drone_coordinate1
                    nearest_drone_coordinate1 = coordinate
                    found_drone = True
                elif distance < min_dist2 and distance != min_dist1:
                    min_dist2 = distance
                    nearest_drone_coordinate2 = coordinate

            if not found_drone:
                return found_drone, command_comm

            # If we found at least 2 drones
            if found_drone and len(received_messages) >= 2:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                (nearest_position2, nearest_angle2) = nearest_drone_coordinate2
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle2 = normalize_angle(
                    nearest_angle2 - self.measured_compass_angle())
                # The mean of 2 angles can be seen as the angle of a vector, which
                # is the sum of the two unit vectors formed by the 2 angles.
                diff_angle = math.atan2(0.5 * math.sin(diff_angle1) + 0.5 * math.sin(diff_angle2),
                                        0.5 * math.cos(diff_angle1) + 0.5 * math.cos(diff_angle2))

            # If we found only 1 drone
            elif found_drone and len(received_messages) == 1:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle = diff_angle1

            # if you are far away, you get closer
            # heading < 0: at left
            # heading > 0: at right
            # base.angular_vel_controller : -1:left, 1:right
            # we are trying to align : diff_angle -> 0
            command_comm["rotation"] = sign(diff_angle)

            # Desired distance between drones
            desired_dist = 60

            d1x = nearest_position1[0] - self.measured_gps_position()[0]
            d1y = nearest_position1[1] - self.measured_gps_position()[1]
            distance1 = math.sqrt(d1x ** 2 + d1y ** 2)

            d1 = distance1 - desired_dist
            # We use a logistic function. -1 < intensity1(d1) < 1 and  intensity1(0) = 0
            # intensity1(d1) approaches 1 (resp. -1) as d1 approaches +inf (resp. -inf)
            intensity1 = 2 / (1 + math.exp(-d1 / (desired_dist * 0.5))) - 1

            direction1 = math.atan2(d1y, d1x)
            heading1 = normalize_angle(direction1 - self.measured_compass_angle())

            # The drone will slide in the direction of heading
            longi1 = intensity1 * math.cos(heading1)
            lat1 = intensity1 * math.sin(heading1)

            # If we found only 1 drone
            if found_drone and len(received_messages) == 1:
                command_comm["forward"] = longi1
                command_comm["lateral"] = lat1

            # If we found at least 2 drones
            elif found_drone and len(received_messages) >= 2:
                d2x = nearest_position2[0] - self.measured_gps_position()[0]
                d2y = nearest_position2[1] - self.measured_gps_position()[1]
                distance2 = math.sqrt(d2x ** 2 + d2y ** 2)

                d2 = distance2 - desired_dist
                intensity2 = 2 / (1 + math.exp(-d2 / (desired_dist * 0.5))) - 1

                direction2 = math.atan2(d2y, d2x)
                heading2 = normalize_angle(direction2 - self.measured_compass_angle())

                longi2 = intensity2 * math.cos(heading2)
                lat2 = intensity2 * math.sin(heading2)

                command_comm["forward"] = 0.5 * (longi1 + longi2)
                command_comm["lateral"] = 0.5 * (lat1 + lat2)

        return found_drone, command_comm
