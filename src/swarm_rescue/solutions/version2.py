import math
from copy import deepcopy
from typing import Optional

import numpy as np
import heapq
import random

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign

from solutions.grid import GridMap, a_star



class MyDroneV2(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=False,
                         **kwargs)
        self.grid = GridMap(size=misc_data.size_area, resolution=25)
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
        found_rescue_center, command_semantic, rescue_center_points = self.process_semantic_sensor(self.semantic())
        found, command_comm = self.process_communication_sensor()

        walls.extend(rescue_center_points)

        alpha = 0.4
        alpha_rot = 0.75

        gps_pos = self.measured_gps_position()


        grid_pos = self.grid.gps_to_grid_cell(gps_pos)

        
        if collision_lidar:
            alpha_rot = 0.1

        """
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
        """

        

        
        
        if ((np.isnan(gps_pos[0]) or np.isnan(gps_pos[1])) == False):
            # Mettre à jour le danger de la cellule actuelle
            self.grid.update_cell_danger(grid_pos[0], grid_pos[1], danger=-1)


            # Si le drone n'a pas encore de cible, il en choisit une au hasard
            while(self.target_cell is None or self.path_to_target is None):
                self.target_cell = self.grid.random_cell_weighted_by_danger()
                self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1])


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
                target_direction = np.array(self.grid.grid_cell_to_gps(self.path_to_target[0])) - np.array(gps_pos)
                target_direction_norm = target_direction / np.linalg.norm(target_direction)
                angle_to_target = np.arctan2(target_direction_norm[1], target_direction_norm[0])

                # Calcul de l'intensité de la rotation en fonction de l'angle uniquement
                angle_diff = angle_to_target - self.measured_compass_angle()[0]
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

                command["forward"] = min(1, 0.2 + np.linalg.norm(target_direction)/200)
                print(command["forward"])
                if np.linalg.norm(target_direction) < ( (self.grid.size[0] / self.grid.cols) / 2):
                    self.path_to_target.pop(0)

            else:
                while(True):
                    self.target_cell = self.grid.random_cell_weighted_by_danger()
                    self.path_to_target = a_star(self.grid, grid_pos[0], grid_pos[1], self.target_cell[0], self.target_cell[1])
                    if not (self.path_to_target is None):
                        break
                    

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
                if values[i] < 200:
                    x = self.measured_gps_position()[0] + values[i] * np.cos(ray_angles[i] + self.measured_compass_angle())
                    y = self.measured_gps_position()[1] + values[i] * np.sin(ray_angles[i] + self.measured_compass_angle())
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
    


    def process_semantic_sensor(self, the_semantic_sensor):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0
    
        detection_semantic = the_semantic_sensor.get_sensor_values()
        best_angle = 1000
    
        found_rescue_center = False
        is_near = False
        rescue_center_points = []
        if detection_semantic:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    best_angle = data.angle
                    is_near = (data.distance < 30)
                    x = self.measured_gps_position()[0] + data.distance * np.cos(data.angle + self.measured_compass_angle())
                    y = self.measured_gps_position()[1] + data.distance * np.sin(data.angle + self.measured_compass_angle())
                    rescue_center_points.append((x, y))
    
        """
        if found_rescue_center:
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
    
        if found_rescue_center and is_near:
            command["forward"] = 0
            command["rotation"] = random.uniform(0.1, 1)
        """
    
        return found_rescue_center, command, rescue_center_points
    


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
