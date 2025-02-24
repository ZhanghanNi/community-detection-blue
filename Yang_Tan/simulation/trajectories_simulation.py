import numpy as np
import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Patch

class AreaSplitter:
    def __init__(self, bounds, n):
        self.bounds = bounds  # (x_min, y_min, x_max, y_max)
        self.n = n
        self.areas = []
        self.vor = None
        self.generate_voronoi_cells()
    
    def voronoi_finite_polygons_2d(self, vor, radius=None):
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        if radius is None:
            radius = np.ptp(vor.points, axis = 0).max() * 2

        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                t = vor.points[p2] - vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)
    
    def generate_voronoi_cells(self):
        x_min, y_min, x_max, y_max = self.bounds
        points = np.random.uniform((x_min, y_min), (x_max, y_max), (self.n, 2))
        self.vor = Voronoi(points)
        bounding_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
        self.areas = []
        
        regions, vertices = self.voronoi_finite_polygons_2d(self.vor)
        for region in regions:
            polygon = Polygon(vertices[region])
            clipped_polygon = polygon.intersection(bounding_polygon)
            if not clipped_polygon.is_empty:
                self.areas.append(clipped_polygon)
    
    def plot_voronoi(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        x_min, y_min, x_max, y_max = self.bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        colors = plt.cm.get_cmap("tab20", len(self.areas))
        print(f"Number of areas: {len(self.areas)}")
        for i, area in enumerate(self.areas):
            if isinstance(area, Polygon):
                x, y = area.exterior.xy
                ax.fill(x, y, color=colors(i), alpha=0.4)
        
        plt.show()
        
    def get_areas(self):
        return self.areas



class Area:
    def __init__(self, bounds):
        self.bounds = bounds
    
    def contains(self, point):
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= point.x <= x_max and y_min <= point.y <= y_max

class Trajectory:
    def __init__(self, moving_object, h, step_size, a, b):
        self.moving_object = moving_object
        self.points = []
        self.h = h
        self.step_size = step_size
        self.a = a
        self.b = b
        self.max_attempts = 5000  # Prevent infinite loops
        self.generate_trajectory()
    
    def move(self, point):
        angle = random.uniform(0, 2 * np.pi)
        circle_x = point.x + self.step_size * np.cos(angle)
        circle_y = point.y + self.step_size * np.sin(angle)
        new_point = Point(circle_x, circle_y)
        # 10% chance to move anywhere in the circle
        if random.random() < 0.1:
            for _ in range(self.max_attempts):
                if 0 <= new_point.x <= self.a and 0 <= new_point.y <= self.b:
                    return new_point  # Found a valid 
             # Generate a new random point
            angle = random.uniform(0, 2 * np.pi)
            circle_x = point.x + self.step_size * np.cos(angle)
            circle_y = point.y + self.step_size * np.sin(angle)
            new_point = Point(circle_x, circle_y)


        for _ in range(self.max_attempts):
            for area in self.moving_object.assigned_areas:
                if area.contains(new_point):
                    return new_point  # Found a valid move
            
            # Generate a new random point
            angle = random.uniform(0, 2 * np.pi)
            circle_x = point.x + self.step_size * np.cos(angle)
            circle_y = point.y + self.step_size * np.sin(angle)
            new_point = Point(circle_x, circle_y)

        # If no valid point is found, return the original point as a fallback
        return point  
    
    def generate_trajectory(self):
        current_point = self.moving_object.random_point_in_area()
        self.points.append(current_point)
        for _ in range(self.h - 1):
            current_point = self.move(current_point)
            self.points.append(current_point)

class MovingObject:
    def __init__(self, assigned_areas, k, step_size, a, b):
        self.assigned_areas = assigned_areas  # Now stores a list of areas
        self.trajectory = Trajectory(self, k, step_size, a, b)
    
    def random_point_in_area(self):
        area = random.choice(self.assigned_areas)
        x_min, y_min, x_max, y_max = area.bounds
        return Point(random.uniform(x_min, x_max), random.uniform(y_min, y_max))

def create_study_area(a=100, b=100, n=2, m=20, k=50, step_size=float('inf')):
    num = 2 * n
    splitter = AreaSplitter((0, 0, a, b), num)
    sub_areas = splitter.get_areas()
    
    if len(sub_areas) < num:
        raise ValueError(f"Expected {num} areas, but got {len(sub_areas)}. Try increasing area splitting precision.")

    moving_objects = []


    for i in range(3 * n):
        # print(f"cluster{i}:")
        if i < num:
            assigned_area = [sub_areas[i]]
        else:
            assigned_area = [sub_areas[2 * (i  % num)], sub_areas[2 * (i  % num)+1]]
        
        for _ in range(m):
            moving_obj = MovingObject(assigned_area, k, step_size, a, b)
            moving_objects.append(moving_obj)
            # print(f"object{_}:")
    
    return sub_areas, moving_objects

def visualize(moving_objects, sub_areas, a, b):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, a)
    ax.set_ylim(0, b)
    colors = plt.cm.get_cmap("Paired", len(sub_areas) * 2)
    area_colors = {}
    
    # Plot each sub-area (Voronoi cell) as its polygonal shape
    for i, area in enumerate(sub_areas):
        area_colors[tuple(area.bounds)] = colors(i)
        
        # Ensure each area is a valid polygon
        if isinstance(area, Polygon):
            x, y = area.exterior.xy
            ax.fill(x, y, color=colors(i), alpha=0.3, label=f'Area {i+1}')
    
    # Plot moving objects with labels for their assigned areas
    for obj in moving_objects:
        color = colors(sub_areas.index(obj.assigned_areas[0])) if len(obj.assigned_areas) == 1 \
            else colors(len(sub_areas) + sub_areas.index(obj.assigned_areas[0]))
        x_vals, y_vals = zip(*[(p.x, p.y) for p in obj.trajectory.points])
        ax.scatter(x_vals, y_vals, s=5, color=color, alpha=0.6)

        # Create label for assigned areas, either 1 or 2 areas
        label = ', '.join([f'Assigned Area {sub_areas.index(area) + 1}' for area in obj.assigned_areas])
        ax.scatter(x_vals[-1], y_vals[-1], s=5, color=color, alpha=0.6, label=label)
    
    # Add the legend for moving objects only
    # We ensure only unique labels are shown by passing unique labels in `handles` and `labels`
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

   # Adjust the legend position outside the plot but within the window
    ax.legend(handles=unique_handles, labels=unique_labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.tight_layout(pad=2)
    plt.show()



