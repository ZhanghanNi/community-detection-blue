import trajectories_simulation as ts
import numpy as np
import networkx as nx
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.colors as mcolors
class Graph_Trajectory_Generator:
    def __init__(self, moving_objects, k_clusters):
        self.moving_objects = moving_objects
        self.k_clusters = k_clusters
        self.graph = nx.Graph()
        self.merged_graph = nx.Graph()
        self.points = []
        self.relationship_matrix = None
        self.generate_graph()
        self.communities, self.point_to_cluster= self.cluster_points()
        self.create_merged_graph()
    
    def generate_graph(self):
        index_map = {}
        index_counter = 0
        
        for obj in self.moving_objects:
            trajectory = obj.trajectory.points
            self.points.extend(trajectory)
            
            for i, point in enumerate(trajectory):
                self.graph.add_node(index_counter, pos=(point.x, point.y))
                index_map[(obj, i)] = index_counter
                index_counter += 1
            
            # Create a complete graph within each trajectory
            for i in range(len(trajectory)):
                for j in range(i + 1, len(trajectory)):
                    self.graph.add_edge(index_map[(obj, i)], index_map[(obj, j)])
        
        self.relationship_matrix = np.zeros((len(self.points), len(self.points)))
        for i, (u, v) in enumerate(self.graph.edges()):
            self.relationship_matrix[u][v] = 1
            self.relationship_matrix[v][u] = 1
     
    def cluster_points(self):
        coords = np.array([(p.x, p.y) for p in self.points])
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42).fit(coords)
        self.clustered_points = kmeans.cluster_centers_

        communities = [[] for _ in range(self.k_clusters)]
        point_to_cluster = {}
        for idx, cluster in enumerate(kmeans.labels_):
            point = self.points[idx]
            communities[cluster].append(point)
            point_to_cluster[point] = cluster

        return communities, point_to_cluster
    
    
    def create_merged_graph(self):
        # Add each cluster's centroid as a node in the merged graph.
        for i, (x, y) in enumerate(self.clustered_points):
            self.merged_graph.add_node(i, pos=(x, y))
        
        # Instead of iterating over communities and calling self.points.index(), we iterate over
        # the original graph’s edges. For each edge, we use point_to_cluster for constant‐time lookup
        # of the cluster index for its endpoints.
        for u, v in self.graph.edges():
            cluster_u = self.point_to_cluster[self.points[u]]
            cluster_v = self.point_to_cluster[self.points[v]]
            # Only process edges that connect different clusters.
            if cluster_u != cluster_v:
                # Order the cluster indices for consistency.
                i, j = sorted((cluster_u, cluster_v))
                if self.merged_graph.has_edge(i, j):
                    self.merged_graph[i][j]['weight'] += self.relationship_matrix[u][v]
                else:
                    self.merged_graph.add_edge(i, j, weight=self.relationship_matrix[u][v])
                # (Optional: you can remove or adjust the debug print if needed.)
                # print(f"Added/updated edge between clusters {i} and {j}")
                
    def get_results(self):
        return self.clustered_points, self.graph, self.merged_graph

def visualize_graphs(original_graph, merged_graph, sub_areas, a, b):
    """
    Visualize the original trajectory graph and the merged graph side by side with subarea polygons.
    
    Parameters:
      original_graph (networkx.Graph): The original trajectory graph.
      merged_graph (networkx.Graph): The merged (clustered) graph.
      sub_areas (list): List of sub-area objects (as shapely Polygons).
      a (int/float): Maximum x value of the rectangle.
      b (int/float): Maximum y value of the rectangle.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # Use the same colormap as before.
    colors = plt.cm.get_cmap("Paired", len(sub_areas) * 2)
    
    # Create a mapping for area colors if needed.
    area_colors = {}
    for i, area in enumerate(sub_areas):
        area_colors[tuple(area.bounds)] = colors(i)
    
    # ----- Original Trajectory Graph -----
    ax_orig = axes[0]
    ax_orig.set_xlim(0, a)
    ax_orig.set_ylim(0, b)
    pos_orig = nx.get_node_attributes(original_graph, 'pos')
    edges_orig = list(original_graph.edges())
    
    # Draw subarea polygons on the original graph with a low zorder.
    for i, area in enumerate(sub_areas):
        if hasattr(area, 'exterior'):
            x, y = area.exterior.xy
            ax_orig.fill(x, y, color=colors(i), alpha=0.3, label=f'Area {i+1}', zorder=1)
    
    # Then draw the edges and nodes on top.
    for edge in edges_orig:
        x_values, y_values = zip(*[pos_orig[edge[0]], pos_orig[edge[1]]])
        ax_orig.plot(x_values, y_values, color='gray', linewidth=0.5, zorder=2)
    ax_orig.scatter(*zip(*pos_orig.values()), color='blue', s=10, zorder=3)
    
    ax_orig.set_title('Original Trajectory Graph')
    ax_orig.set_aspect('equal', adjustable='box')
    
    # ----- Merged (Clustered) Graph -----
    ax_merged = axes[1]
    ax_merged.set_xlim(0, a)
    ax_merged.set_ylim(0, b)
    pos_merged = nx.get_node_attributes(merged_graph, 'pos')
    edges_merged = list(merged_graph.edges())
    
    # Draw subarea polygons on the merged graph with the same settings.
    for i, area in enumerate(sub_areas):
        if hasattr(area, 'exterior'):
            x, y = area.exterior.xy
            ax_merged.fill(x, y, color=colors(i), alpha=0.3, label=f'Area {i+1}', zorder=1)
    
    # Then draw the merged graph edges and nodes on top.
    for edge in edges_merged:
        x_values, y_values = zip(*[pos_merged[edge[0]], pos_merged[edge[1]]])
        ax_merged.plot(x_values, y_values, color='black', linewidth=0.5, zorder=2)
    ax_merged.scatter(*zip(*pos_merged.values()), color='green', s=20, zorder=3)
    
    ax_merged.set_title('Merged Graph (Clustered Points)')
    ax_merged.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

    
def main(benchmarking_mode = False) -> nx.Graph:
    # Set parameters
    a = 100  # x-axis range
    b = 100  # y-axis range
    n = 3  # number of sub-areas
    m = 10  # number of points per sub-area
    h = 10  # steps in each trajectory
    step_size = 50  # maximum step size for trajectories
    
    ratio = 0.1
    k = int((3 * n * m * h) * ratio)

    # Create study area and generate points
    sub_areas, moving_objects = ts.create_study_area(a, b, n, m, h, step_size)
    if not benchmarking_mode:
        ts.visualize(moving_objects, sub_areas, a, b)
    
     # Generate graphs from moving_objects using our Graph_Trajectory_Generator
    graph_generator = Graph_Trajectory_Generator(moving_objects, k)
    
    if not benchmarking_mode:
        # Visualize both graphs with subareas overlaid
        visualize_graphs(graph_generator.graph, graph_generator.merged_graph, sub_areas, a, b)
    
    return graph_generator.get_results()[2]

if __name__ == "__main__":
    main()