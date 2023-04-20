import heapq
import random
import math

import numpy as np



class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
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
        self.rescue_centers = []  # nouvelle liste pour les centres de sauvetage
        
        # Ajout des bordures de la carte
        for i in range(self.rows):
            if i == 0 or i == self.rows-1:
                for j in range(self.cols):
                    self.map[i][j] = 1
            else:
                self.map[i][0] = 1
                self.map[i][self.cols-1] = 1
        
        print(self.rows, self.cols)

    def add_rescue_center(self, x, y):
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return
        self.rescue_centers.append((x, y))

    def update_wall(self, x, y):
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return
        if (self.map[x][y] != -1):
            self.map[x][y] = 1

    def update_cell_danger(self, x, y, danger, override=False):
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return
        if self.map[x][y] in [1, -1] and override==False:
            return
        for i in range(max(0, x-5), min(x+6, self.rows)):
            for j in range(max(0, y-5), min(y+6, self.cols)):
                if i == x and j == y:
                    self.map[i][j] = danger
                else:
                    dist = max(abs(i-x), abs(j-y))
                    self.map[i][j] = max(min(self.map[i][j] + (danger * ((1 - abs(self.map[i][j])) / (dist + 1))), 1), -1)

    """
    def random_cell_weighted_by_danger(self):
        # Création d'une liste contenant toutes les cellules de la grille avec leur poids respectif
        cells = []
        for i in range(self.rows):
            for j in range(self.cols):
                weight = (1 - abs(self.map[i][j]))**3
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
    """

    def random_cell_weighted_by_danger(self):
        # Création d'une liste contenant toutes les cellules de la grille avec leur poids respectif
        cells = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.map[i][j] == -1:
                    continue
                weight_sum = 0
                count = 0
                for x in range(max(1, i-4), min(i+5, self.rows-1)):
                    for y in range(max(1, j-4), min(j+5, self.cols-1)):
                            weight_sum += 1 - abs(self.map[x][y])
                            count += 1
                if count > 0:
                    avg_weight = weight_sum / count
                else:
                    avg_weight = 0
                weight = avg_weight**5
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
            return (min(max(0, row+1), self.rows-1), min(max(0, col+1), self.cols-1))
    
    def grid_cell_to_gps(self, cell_pos):
        """
        Convertit des indices de cellules dans la représentation sous forme de grille en coordonnées GPS.
        """
        row, col = cell_pos
        x = ((row-1) * self.resolution) - (self.size[0] / 2) + (self.resolution / 2)
        y = ((col-1) * self.resolution) - (self.size[1] / 2) + (self.resolution / 2)
        return (x, y)

    
import math

def closest_rescue_center_index(rescue_centers, x, y):
    if len(rescue_centers) == 0:
        return None
        
    # Calcul de la distance entre chaque centre de sauvetage et la cellule donnée
    distances = []
    for center in rescue_centers:
        dist = math.sqrt((center[0]-x)**2 + (center[1]-y)**2)
        distances.append(dist)
        
    # Renvoi de l'indice du centre de sauvetage le plus proche
    index_min = distances.index(min(distances))
    return index_min




def a_star(grid_map, start_x, start_y, end_x, end_y):
    # Création des noeuds de départ et d'arrivée
    start_node = Node(start_x, start_y)
    end_node = Node(end_x, end_y)
    
    # Initialisation de la liste ouverte et de l'ensemble fermé
    open_list = [start_node]
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
        for neighbor_x, neighbor_y in [(current_node.x-1, current_node.y), (current_node.x+1, current_node.y), (current_node.x, current_node.y-1), (current_node.x, current_node.y+1)]:
            # Vérification que le voisin est dans la grille
            if neighbor_x < 0 or neighbor_x >= grid_map.rows or neighbor_y < 0 or neighbor_y >= grid_map.cols:
                continue
            # Vérification que le voisin n'est pas un obstacle
            if grid_map.map[neighbor_x][neighbor_y] == 1 and (neighbor_x != end_node.x or neighbor_y != end_node.y) :
                continue
            # Vérification que le voisin n'a pas déjà été exploré
            if (neighbor_x, neighbor_y) in closed_set:
                continue
            
            # Création d'un nouveau noeud pour le voisin
            neighbor_node = Node(neighbor_x, neighbor_y)
            # Calcul du coût de déplacement depuis le noeud courant
            neighbor_node.g = current_node.g + 1 + (1 + grid_map.map[neighbor_x][neighbor_y])**2
            # Calcul de la valeur heuristique pour estimer le coût restant jusqu'à l'arrivée
            neighbor_node.h = 1 + min([grid_map.map[nx][ny] for nx, ny in [(neighbor_x-1, neighbor_y), (neighbor_x+1, neighbor_y), (neighbor_x, neighbor_y-1), (neighbor_x, neighbor_y+1)] if 0 <= nx < grid_map.rows and 0 <= ny < grid_map.cols])
            # Calcul du coût total (f)
            neighbor_node.f = neighbor_node.g + (1 + neighbor_node.h**2) * (abs(end_x-neighbor_x) + abs(end_y-neighbor_y))
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
    
    # Si on n'a pas trouvé de chemin, on renvoie None
    return None



def a_star_rescue(grid_map, start_x, start_y, end_x, end_y):
    # Création des noeuds de départ et d'arrivée
    start_node = Node(start_x, start_y)
    end_node = Node(end_x, end_y)
    
    # Initialisation de la liste ouverte et de l'ensemble fermé
    open_list = [start_node]
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
        for neighbor_x, neighbor_y in [(current_node.x-1, current_node.y), (current_node.x+1, current_node.y), (current_node.x, current_node.y-1), (current_node.x, current_node.y+1)]:
            # Vérification que le voisin est dans la grille
            if neighbor_x < 0 or neighbor_x >= grid_map.rows or neighbor_y < 0 or neighbor_y >= grid_map.cols:
                continue
            # Vérification que le voisin a une valeur de danger égale à -1
            if grid_map.map[neighbor_x][neighbor_y] > -0.85 and (neighbor_x != end_node.x or neighbor_y != end_node.y):
                continue
            # Vérification que le voisin n'a pas déjà été exploré
            if (neighbor_x, neighbor_y) in closed_set:
                continue
            
            # Création d'un nouveau noeud pour le voisin
            neighbor_node = Node(neighbor_x, neighbor_y)
            # Calcul du coût de déplacement depuis le noeud courant
            neighbor_node.g = current_node.g + 1
            # Calcul de la valeur heuristique pour estimer le coût restant jusqu'à l'arrivée
            neighbor_node.h = abs(end_x-neighbor_x) + abs(end_y-neighbor_y)
            # Calcul du coût total (f)
            neighbor_node.f = neighbor_node.g + neighbor_node.h
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
    
    # Si on n'a pas trouvé de chemin, on renvoie None
    return None