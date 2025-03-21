import trajectories_simulation as ts
import trajectory_graph_generator as tg
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

import sys
sys.path.append("../../Aidan/girvan_newman_implementation")
import girvan_newman_implementation

class Result_Analysis:
    def __init__(self, graph, communities, subareas, a, b, algorithm_name):
        self.graph = graph
        self.communities = communities  # List of lists of node indices
        self.subareas = [Polygon(subarea) for subarea in subareas]  # Convert to Shapely polygons
        self.a = a
        self.b = b
        self.algorithm_name = algorithm_name
        
    def voronoi_finite_polygons_2d(self, vor, radius=None):
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        if radius is None:
            radius = np.ptp(vor.points, axis=0).max() * 2

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
    
    def generate_voronoi_polygons(self):
        positions = np.array([self.graph.nodes[n]['pos'] for n in self.graph.nodes])
        vor = Voronoi(positions)
        
        bounding_polygon = Polygon([(0, 0), (self.a, 0), (self.a, self.b), (0, self.b)])
        
        areas = {}
        regions, vertices = self.voronoi_finite_polygons_2d(vor)
        for idx, region in enumerate(regions):
            polygon = Polygon(vertices[region])
            clipped_polygon = polygon.intersection(bounding_polygon)
            if not clipped_polygon.is_empty:
                areas[idx] = clipped_polygon
        
        return areas
    
    def plot_results(self):
        areas = self.generate_voronoi_polygons()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1 = axes[0]
        cmap = plt.get_cmap("tab10")
        node_to_color = {}
        
        # Assign colors based on community
        for i, community in enumerate(self.communities):
            for node in community:
                node_to_color[node] = cmap(i % 10)
                
        # Plot Voronoi regions
        for node_idx, area in areas.items():
            x, y = area.exterior.xy
            color = node_to_color.get(node_idx, 'gray')
            ax1.fill(x, y, alpha=0.5, edgecolor='black', facecolor=color)
        
        # Plot community nodes
        for node, color in node_to_color.items():
            pos = self.graph.nodes[node]['pos']
            ax1.scatter(*pos, color=color, edgecolor='black')

        ax1.set_xlim(0, self.a)
        ax1.set_ylim(0, self.b)
        ax1.set_title(f"Clipped Voronoi Polygons Labeled with Communities Generated by {self.algorithm_name}")

        # Plot the subareas on the second axis
        ax2 = axes[1]
        patches = [MplPolygon(list(subarea.exterior.coords), closed=True) for subarea in self.subareas]
        p = PatchCollection(patches, alpha=0.4, facecolor='blue', edgecolor='black')
        ax2.add_collection(p)

        ax2.set_xlim(0, self.a)
        ax2.set_ylim(0, self.b)
        ax2.set_title("Subareas")
        
        plt.tight_layout()
        plt.show()


def main():
    # Set parameters
    a = 100  # x-axis range
    b = 100  # y-axis range
    n = 3  # number of sub-areas
    m = 20  # number of points per sub-area
    h = 10  # steps in each trajectory
    step_size = 20  # maximum step size for trajectories
    
    ratio = 0.4
    k = int((3 * n * m * h) * ratio)

    # Create study area and generate points
    sub_areas, moving_objects = ts.create_study_area(a, b, n, m, h, step_size)
    ts.visualize(moving_objects, sub_areas, a, b)
    
     # Generate graphs from moving_objects using our Graph_Trajectory_Generator
    graph_generator = tg.Graph_Trajectory_Generator(moving_objects, k)
    original_graph = graph_generator.graph 
    merged_graph = graph_generator.merged_graph
    # Visualize both graphs with subareas overlaid
    tg.visualize_graphs(original_graph, merged_graph, sub_areas, a, b)
    
    
    
    '''
    Run the algorithm here!
    '''

    communities = girvan_newman_implementation.main(merged_graph, weighted=nx.is_weighted(merged_graph))
    ra = Result_Analysis(merged_graph, communities, sub_areas, a, b)
    ra.plot_results()
    
    
    
if __name__ == "__main__":
    main()
