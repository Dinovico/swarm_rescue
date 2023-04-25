import math
import random
from copy import deepcopy
from typing import Optional
from enum import Enum

import numpy as np


from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign

from solutions.gridV2 import GridMap, a_star, a_star_rescue, closest_rescue_center_index, merge_grid






class MyDroneV6(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=False,
                         **kwargs)
        self.grid = GridMap(size=misc_data.size_area, resolution=15)
        self.target_cell = None  # La cellule cible que le drone doit atteindre
        self.path_to_target = []  # Le chemin à suivre pour atteindre la cible
        # The state is initialized to searching wounded person
        self.state = self.Activity.EXPLORING
        self.currently_rescuing = False
        self.step_count = 0
        self.last_position = None
        self.last_angle = None

    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        
        if(self.step_count > 1):
            msg_data = (self.identifier, self.grid, self.grid.gps_to_grid_cell(self.last_position))
            return msg_data

        return None


    def control(self):
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        self.step_count += 1

        print(self.identifier)

        
        if self.identifier == 4:
            return self.control_master() 
        

        
        ##########
        # COMMANDS FOR EACH STATE
        ##########
        if self.state is self.Activity.EXPLORING:
            found_wounded, command = self.control_explore()

        elif self.state is self.Activity.GRASPING_WOUNDED:
            found_wounded, command = self.control_grasp()

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            found_wounded, command = self.control_rescue()


        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.EXPLORING and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = self.state.DROPPING_AT_RESCUE_CENTER
            self.currently_rescuing = False
            print(self.identifier)
            print(self.grid.map)

        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.state.EXPLORING

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.state = self.Activity.EXPLORING
            self.path_to_target = None
            self.currently_rescuing = False
            

        

        return command

    def control_master(self):

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        if(self.step_count <= 2):
            self.last_position = self.measured_gps_position()
            self.last_angle = self.measured_compass_angle()

        else:
            gps_pos = self.measured_gps_position()
            angle = self.measured_compass_angle()

            grid_pos = self.grid.gps_to_grid_cell(gps_pos)

            self.grid.update_cell_danger(grid_pos[0], grid_pos[1], -1, override=True)

            target_direction = np.array(self.last_position) - np.array(gps_pos)
            target_direction_norm = target_direction / np.linalg.norm(target_direction)
            angle_to_target = np.arctan2(target_direction_norm[1], target_direction_norm[0])

            angle_diff = angle_to_target - angle
            if angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            elif angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            rotation_intensity = 0.3 * sign(angle_diff) + angle_diff / np.pi


            if rotation_intensity > 0:
                command["rotation"] = min(rotation_intensity, 1)
            else:
                command["rotation"] = max(rotation_intensity, -1)

            command["forward"] = min(1, np.linalg.norm(target_direction)/100)


        self.process_communication_sensor()

        return command




    def control_explore(self):

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}



        if self.gps_is_disabled():
            # Récupération des données de l'odomètre
            odometer_values = self.odometer_values()
            dist_travel, alpha, theta = odometer_values[0], odometer_values[1], odometer_values[2]
            

            # Interpolation de la position actuelle avec les valeurs de l'odomètre
            self.last_position = self.last_position + np.array([dist_travel * np.cos(self.last_angle[0]), dist_travel * np.sin(self.last_angle[0])])
            self.last_angle = normalize_angle(self.last_angle + theta)

        
        else:
            self.last_position = self.measured_gps_position()

            self.last_angle = self.measured_compass_angle()


        command_lidar, collision_lidar, walls = self.process_lidar_sensor(
            self.lidar())
        found_wounded, found_rescue_center, command_semantic, rescue_center_points, wounded_person_points = self.process_semantic_sensor(self.semantic())
        command_comm, drone_cells = self.process_communication_sensor()

        grid_pos = self.grid.gps_to_grid_cell(self.last_position)

        n = len(drone_cells)
        for i in range(n):
            drone_cells.append((drone_cells[i][0]+1, drone_cells[i][1]))
            drone_cells.append((drone_cells[i][0]+1, drone_cells[i][1]+1))
            drone_cells.append((drone_cells[i][0]+1, drone_cells[i][1]-1))
            drone_cells.append((drone_cells[i][0]-1, drone_cells[i][1]))
            drone_cells.append((drone_cells[i][0]-1, drone_cells[i][1]+1))
            drone_cells.append((drone_cells[i][0]-1, drone_cells[i][1]-1))
            drone_cells.append((drone_cells[i][0], drone_cells[i][1]+1))
            drone_cells.append((drone_cells[i][0], drone_cells[i][1]-1))
            
        for i in range(len(drone_cells)):
            self.grid.update_cell_danger(drone_cells[i][0], drone_cells[i][1], -1)


        if(self.step_count <= 2):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    self.grid.update_cell_danger(grid_pos[0]+i, grid_pos[1]+j, danger=-1, override=True)

        
        
        if ((np.isnan(self.last_position[0]) or np.isnan(self.last_position[1])) == False):
            # Mettre à jour le danger de la cellule actuelle
            self.grid.update_cell_danger(grid_pos[0], grid_pos[1], danger=-1)


            # Si le drone n'a pas encore de cible, il en choisit une au hasard
            while(self.target_cell is None or self.path_to_target is None):
                self.target_cell = self.grid.random_cell_weighted_by_danger(((2 * (self.identifier % 2) + 1) * (self.grid.rows // 4), (2 * (self.identifier // 2) + 1) * (self.grid.cols // 4)), self.grid.rows // 4, self.grid.cols // 4)
                while(abs(self.target_cell[0] - grid_pos[0]) + abs(self.target_cell[1] - grid_pos[1]) > max(self.grid.rows // 4, self.grid.cols // 4)):
                    self.target_cell = ((self.target_cell[0] + grid_pos[0]) // 2, (self.target_cell[1] + grid_pos[1]) // 2)
                k = 0
                while(self.path_to_target is None and k < 30):
                    self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1], (abs(grid_pos[0] - self.target_cell[0]) + abs(grid_pos[1] - self.target_cell[1]) * 2 + k))
                    k += 1


            if (len(walls) > 0):
                for i in range(len(walls)):
                    wall_cell = self.grid.gps_to_grid_cell(walls[i])
                    if (self.grid.map[wall_cell[0]][wall_cell[1]] != 1 
                            and wall_cell not in wounded_person_points
                            and (wall_cell[0]+1,wall_cell[1]) not in wounded_person_points
                            and (wall_cell[0]-1,wall_cell[1]) not in wounded_person_points
                            and (wall_cell[0],wall_cell[1]+1) not in wounded_person_points
                            and (wall_cell[0],wall_cell[1]-1) not in wounded_person_points
                            and wall_cell not in drone_cells):
                        self.grid.update_wall(wall_cell[0], wall_cell[1])

                for i in range(len(walls)):
                    wall_cell = self.grid.gps_to_grid_cell(walls[i])
                    if (wall_cell in self.path_to_target and wall_cell not in drone_cells):
                        if (wall_cell == self.target_cell):
                            self.target_cell = None
                            self.path_to_target = None
                            break
                        index_of_wall = self.path_to_target.index(wall_cell)
                        new_path = self.path_to_target[index_of_wall+2:]
                        next_cell = self.path_to_target[index_of_wall+1]
                        while(True):
                            k = 0
                            self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], next_cell[0], next_cell[1], (abs(grid_pos[0] - next_cell[0]) + abs(grid_pos[1] - next_cell[1]) + k))
                            while(self.path_to_target is None and k < 30):
                                k += 1
                                self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], next_cell[0], next_cell[1], (abs(grid_pos[0] - next_cell[0]) + abs(grid_pos[1] - next_cell[1]) + k))
                            if not (self.path_to_target is None):
                                break
                            self.target_cell = self.grid.random_cell_weighted_by_danger(((2 * (self.identifier % 2) + 1) * (self.grid.rows // 4), (2 * (self.identifier // 2) + 1) * (self.grid.cols // 4)), self.grid.rows // 4, self.grid.cols // 4)
                            while(abs(self.target_cell[0] - grid_pos[0]) + abs(self.target_cell[1] - grid_pos[1]) > max(self.grid.rows // 4, self.grid.cols // 4)):
                                self.target_cell = ((self.target_cell[0] + grid_pos[0]) // 2, (self.target_cell[1] + grid_pos[1]) // 2)
                            next_cell = self.target_cell
                            new_path = []
                        self.path_to_target = self.path_to_target + new_path
            

            if (len(rescue_center_points) > 0):
                for i in range(len(rescue_center_points)):
                    rescue_center_cell = self.grid.gps_to_grid_cell(rescue_center_points[i])
                    if (rescue_center_cell not in self.grid.rescue_centers):
                        self.grid.add_rescue_center(rescue_center_cell[0], rescue_center_cell[1])
            
            for i in range(len(wounded_person_points)):
                self.grid.update_cell_danger(wounded_person_points[i][0], wounded_person_points[i][1], 0, override=True)


            # Le drone suit le chemin vers sa cible
            if (self.path_to_target is not None) and (len(self.path_to_target) > 0):
                target_direction = np.array(self.grid.grid_cell_to_gps(self.path_to_target[0])) - np.array(self.last_position)
                target_direction_norm = target_direction / np.linalg.norm(target_direction)
                angle_to_target = np.arctan2(target_direction_norm[1], target_direction_norm[0])

                # Calcul de l'intensité de la rotation en fonction de l'angle uniquement
                angle_diff = angle_to_target - self.last_angle
                if angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                elif angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                rotation_intensity = 0.3 * sign(angle_diff) + angle_diff / np.pi


                # Affectation de la commande de rotation en fonction de l'intensité calculée
                if rotation_intensity > 0:
                    command["rotation"] = min(rotation_intensity, 1)
                else:
                    command["rotation"] = max(rotation_intensity, -1)

                if self.gps_is_disabled():
                    command["forward"] = min(1, 0.1 + np.linalg.norm(target_direction)/200)
                else:
                    command["forward"] = min(1, 0.3 + np.linalg.norm(target_direction)/200)
                if np.linalg.norm(target_direction) < 50:
                    self.path_to_target.pop(0)


            else:
                while(True):
                    self.target_cell = self.grid.random_cell_weighted_by_danger(((2 * (self.identifier % 2) + 1) * (self.grid.rows // 4), (2 * (self.identifier // 2) + 1) * (self.grid.cols // 4)), self.grid.rows // 4, self.grid.cols // 4)
                    while(abs(self.target_cell[0] - grid_pos[0]) + abs(self.target_cell[1] - grid_pos[1]) > max(self.grid.rows // 4, self.grid.cols // 4)):
                        self.target_cell = ((self.target_cell[0] + grid_pos[0]) // 2, (self.target_cell[1] + grid_pos[1]) // 2)
                    k = 0
                    self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1], (abs(grid_pos[0] - self.target_cell[0]) + abs(grid_pos[1] - self.target_cell[1]) + k//20))
                    while(self.path_to_target is None and k < 30):
                        k += 1
                        self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1], (abs(grid_pos[0] - self.target_cell[0]) + abs(grid_pos[1] - self.target_cell[1]) + k))
                    if not (self.path_to_target is None):
                        break
               
        if(len(self.path_to_target) > (self.grid.rows // 2 + self.grid.cols // 2)):
            self.path_to_target = None

        return found_wounded, command




    






    def control_grasp(self):

        found_wounded, found_rescue_center, command_semantic, rescue_center_points, wounded_person_points = self.process_semantic_sensor(self.semantic())

        if self.gps_is_disabled():
            # Récupération des données de l'odomètre
            odometer_values = self.odometer_values()
            dist_travel, alpha, theta = odometer_values[0], odometer_values[1], odometer_values[2]

            # Interpolation de la position actuelle avec les valeurs de l'odomètre
            self.last_position = self.last_position + np.array([dist_travel * np.cos(self.last_angle[0]), dist_travel * np.sin(self.last_angle[0])])
            self.last_angle = normalize_angle(self.last_angle + theta)

        else:
            self.last_position = self.measured_gps_position()
            self.last_angle = self.measured_compass_angle()

        gps_pos = self.last_position

        grid_pos = self.grid.gps_to_grid_cell(gps_pos)

        self.grid.update_cell_danger(grid_pos[0], grid_pos[1], danger=-1, override=True)

        command = command_semantic
        command["grasper"] = 1

        return found_wounded, command

    






    def control_rescue(self):
    
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
    
        if self.gps_is_disabled():
            # Récupération des données de l'odomètre
            odometer_values = self.odometer_values()
            dist_travel, alpha, theta = odometer_values[0], odometer_values[1], odometer_values[2]
    
            # Interpolation de la position actuelle avec les valeurs de l'odomètre
            self.last_position = self.last_position + np.array([dist_travel * np.cos(self.last_angle[0]), dist_travel * np.sin(self.last_angle[0])])
            self.last_angle = normalize_angle(self.last_angle + theta)
    
        else:
            gps_pos = self.measured_gps_position()
            self.last_position = gps_pos
            self.last_angle = self.measured_compass_angle()

        


        if ((np.isnan(self.last_position[0]) or np.isnan(self.last_position[1])) == False):
    
            self.last_position += np.array([np.cos(self.last_angle[0]) * 15, np.sin(self.last_angle[0]) * 15]) 
            grid_pos = self.grid.gps_to_grid_cell(self.last_position)
        
            if not self.currently_rescuing:
                self.grid.update_cell_danger(grid_pos[0], grid_pos[1], -1, override=True)
        
                not_tried_rescue = self.grid.rescue_centers.copy()
                while(True):
                    
                    index = closest_rescue_center_index(not_tried_rescue, grid_pos[0], grid_pos[1])
                    self.target_cell = not_tried_rescue.pop(index)
                        
                    self.path_to_target = a_star_rescue(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1])
                    if not (self.path_to_target is None):
                            self.currently_rescuing = True
                            break
                    
            command_lidar, collision_lidar, walls = self.process_lidar_sensor(self.lidar())
            found_wounded, found_rescue_center, command_semantic, rescue_center_points, wounded_person_points = self.process_semantic_sensor(self.semantic())
            command_comm, drone_cells = self.process_communication_sensor()
            
            n = len(drone_cells)
            for i in range(n):
                drone_cells.append((drone_cells[i][0]+1, drone_cells[i][1]))
                drone_cells.append((drone_cells[i][0]+1, drone_cells[i][1]+1))
                drone_cells.append((drone_cells[i][0]+1, drone_cells[i][1]-1))
                drone_cells.append((drone_cells[i][0]-1, drone_cells[i][1]))
                drone_cells.append((drone_cells[i][0]-1, drone_cells[i][1]+1))
                drone_cells.append((drone_cells[i][0]-1, drone_cells[i][1]-1))
                drone_cells.append((drone_cells[i][0], drone_cells[i][1]+1))
                drone_cells.append((drone_cells[i][0], drone_cells[i][1]-1))
                self.grid.update_cell_danger(drone_cells[i][0], drone_cells[i][1], -1, override=True)


            if (len(walls) > 0):
                for i in range(len(walls)):
                    wall_cell = self.grid.gps_to_grid_cell(walls[i])
                    if (self.grid.map[wall_cell[0]][wall_cell[1]] != 1 
                            and wall_cell not in wounded_person_points
                            and (wall_cell[0]+1,wall_cell[1]) not in wounded_person_points
                            and (wall_cell[0]-1,wall_cell[1]) not in wounded_person_points
                            and (wall_cell[0],wall_cell[1]+1) not in wounded_person_points
                            and (wall_cell[0],wall_cell[1]-1) not in wounded_person_points
                            and wall_cell not in drone_cells):
                        self.grid.update_wall(wall_cell[0], wall_cell[1])

                for i in range(len(walls)):
                        wall_cell = self.grid.gps_to_grid_cell(walls[i])
                        if ((wall_cell in self.path_to_target) and (wall_cell not in self.grid.rescue_centers) and (wall_cell not in drone_cells)):
                            index_of_wall = self.path_to_target.index(wall_cell)
                            new_path = self.path_to_target[index_of_wall+2:]
                            next_cell = self.path_to_target[index_of_wall+1]
                            not_tried_rescue = self.grid.rescue_centers.copy()
                            while(True):
                                self.path_to_target = a_star_rescue(self.grid, grid_pos[0], grid_pos[1], next_cell[0], next_cell[1])
                                if not (self.path_to_target is None):
                                    break
                                index = closest_rescue_center_index(not_tried_rescue, grid_pos[0], grid_pos[1])
                                self.target_cell = not_tried_rescue.pop(index)
                                next_cell = self.target_cell
                                new_path = []
                            self.path_to_target = self.path_to_target + new_path

            self.grid.update_cell_danger(grid_pos[0], grid_pos[1], -1)

            # Le drone suit le chemin vers sa cible
            if len(self.path_to_target) > 0:
                target_direction = np.array(self.grid.grid_cell_to_gps(self.path_to_target[0])) - np.array(self.last_position)
                target_direction_norm = target_direction / np.linalg.norm(target_direction)
                angle_to_target = np.arctan2(target_direction_norm[1], target_direction_norm[0])
        
                # Calcul de l'intensité de la rotation en fonction de l'angle uniquement
                angle_diff = angle_to_target - self.last_angle[0]
                if angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                elif angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                rotation_intensity = 0.3 * sign(angle_diff) + angle_diff / np.pi
        
        
                # Affectation de la commande de rotation en fonction de l'intensité calculée
                if rotation_intensity > 0:
                    command["rotation"] = min(rotation_intensity, 1)
                else:
                    command["rotation"] = max(rotation_intensity, -1)

                if self.gps_is_disabled():
                    command["forward"] = min(1, 0.1 + np.linalg.norm(target_direction)/200)
                else:
                    command["forward"] = min(1, 0.2 + np.linalg.norm(target_direction)/100)
                if np.linalg.norm(target_direction) < 50:
                    self.path_to_target.pop(0)
        
            else:
                self.currently_rescuing = False
        
            command["grasper"] = 1
    
        return False, command
    







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

        collision_points = []

        
        if size != 0:

            for i in range(size):
                if values[i] < 175:
                    x = self.last_position[0] + values[i] * np.cos(ray_angles[i] + self.last_angle[0])
                    y = self.last_position[1] + values[i] * np.sin(ray_angles[i] + self.last_angle[0])
                    collision_points.append((x, y))


        return command, False, collision_points
    


    def process_semantic_sensor(self, the_semantic_sensor):

        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0
    
        detection_semantic = the_semantic_sensor.get_sensor_values()
    
        found_rescue_center = False
        found_wounded = False

        is_near = False
        rescue_center_points = []
        wounded_person_points = []

        best_angle = 1000
        best_score = 10000



        if detection_semantic:
            scores = []
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    best_angle = data.angle
                    x = self.last_position[0] + data.distance * np.cos(data.angle + self.last_angle[0])
                    y = self.last_position[1] + data.distance * np.sin(data.angle + self.last_angle[0])
                    rescue_center_points.append((x, y))
             

                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))
                    x = self.last_position[0] + data.distance * np.cos(data.angle + self.last_angle[0])
                    y = self.last_position[1] + data.distance * np.sin(data.angle + self.last_angle[0])
                    wounded_person_points.append(self.grid.gps_to_grid_cell((x, y)))

            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]
            
            
    
        
        if self.state is self.Activity.GRASPING_WOUNDED and found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max
    
            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

    
        return found_wounded, found_rescue_center, command, rescue_center_points, wounded_person_points
    

    def process_communication_sensor(self):

        command_comm = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0}

        drone_cells = []

        if(self.step_count > 1):

            if self.communicator:
                received_messages = self.communicator.received_messages

            
        
            if(self.identifier == 9):
                for msg in received_messages:
                    received_id, received_grid, received_pos = msg[1]
                    if(self.step_count % 10 == 0):
                        self.grid = merge_grid(self.grid, received_grid)
            else:
                for msg in received_messages:
                    received_id, received_grid, received_pos = msg[1]
                    drone_cells.append(received_pos)
                    
                    if(received_id == 9):
                        if(self.step_count % 10 == 0):
                            self.grid = merge_grid(self.grid, received_grid) 

        return command_comm, drone_cells

"""
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
"""
