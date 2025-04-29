import osmnx as ox
import rustworkx as rx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import gymnasium as gym
from gymnasium import spaces
import random
import time
import os
import math
import itertools
import scipy
import pickle
import multiprocessing
from datetime import datetime, timedelta
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from learning_analyzer import analyze_learning_curve, find_latest_rewards_data
'''
Implementation using RustworkX instead of NetworkX to
allow faster lengths and connectivity calculations.

Further additions:
1. Cleaner stdout/print statement - only shows progress of the steps
2. Use SubprocVecEnv and multiple envs for parallel training - use max-1 core count
3. Change sampling for connectivity calculation - faster calculation of large graph
4. Group all figures by run in the same subdirectory
'''

# Create a global variable to store the run ID and figures directory
RUN_ID = None
RUN_FIGURES_DIR = None

def generate_run_id():
    """Generate a unique run ID based on timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"

def setup_run_directory(city_name=None):
    """Setup the run-specific directory for all figure outputs"""
    global RUN_ID, RUN_FIGURES_DIR
    
    # Generate a unique run ID
    RUN_ID = generate_run_id()
    
    # Create a base directory for all figure outputs
    base_dir = "./bike_path_figures"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a descriptive name for the run directory
    if city_name:
        safe_city_name = city_name.replace(', ', '_').replace(' ', '_')
        run_dir_name = f"{safe_city_name}_{RUN_ID}"
    else:
        run_dir_name = RUN_ID
    
    # Create the full path to the run directory
    RUN_FIGURES_DIR = os.path.join(base_dir, run_dir_name)
    os.makedirs(RUN_FIGURES_DIR, exist_ok=True)
    
    print(f"All figures for this run will be saved to: {RUN_FIGURES_DIR}")
    return RUN_FIGURES_DIR

# Utility functions for NetworkX to rustworkx conversion
def nx_to_rx(nx_graph, weight_attribute='length'):
    """Convert a NetworkX graph to a rustworkx graph using the built-in converter."""
    # Use the built-in rustworkx converter
    rx_graph = rx.networkx_converter(nx_graph, keep_attributes=True)
    
    # Build a mapping from original node IDs to rustworkx indices
    node_map = {}
    for idx in range(rx_graph.num_nodes()):
        node_data = rx_graph.get_node_data(idx)
        if "__networkx_node__" in node_data:
            original_node = node_data["__networkx_node__"]
            node_map[original_node] = idx
    
    # Ensure weight attribute exists on edges
    if weight_attribute:
        for u_idx in range(rx_graph.num_nodes()):
            for edge_idx, v_idx, data in rx_graph.out_edges(u_idx):
                if weight_attribute not in data:
                    data[weight_attribute] = 1.0
                    rx_graph.update_edge_by_index(edge_idx, data)
    
    return rx_graph, node_map

def rx_to_nx(rx_graph, node_map=None, is_directed=None):
    """Convert a rustworkx graph back to a NetworkX graph."""
    import networkx as nx  # Import here for compatibility with existing code
    
    # Determine if directed if not specified
    if is_directed is None:
        is_directed = isinstance(rx_graph, rx.PyDiGraph)
    
    nx_graph = nx.DiGraph() if is_directed else nx.Graph()
    
    # Add nodes with their attributes
    for idx in range(rx_graph.num_nodes()):
        data = rx_graph.get_node_data(idx)
        
        # Check if the node has the networkx original node ID
        if "__networkx_node__" in data:
            original_node = data["__networkx_node__"]
            # Create a clean copy of the data without the networkx reference
            clean_data = {k: v for k, v in data.items() if k != "__networkx_node__"}
            nx_graph.add_node(original_node, **clean_data)
        else:
            # If no original ID, use the index as the node ID
            nx_graph.add_node(idx, **data)
    
    # Add edges with their attributes
    if isinstance(rx_graph, rx.PyDiGraph):
        # For directed graphs
        for u_idx in range(rx_graph.num_nodes()):
            u_data = rx_graph.get_node_data(u_idx)
            u = u_data.get("__networkx_node__", u_idx)
            
            for _, v_idx, data in rx_graph.out_edges(u_idx):
                v_data = rx_graph.get_node_data(v_idx)
                v = v_data.get("__networkx_node__", v_idx)
                nx_graph.add_edge(u, v, **data)
    else:
        # For undirected graphs
        for u_idx in range(rx_graph.num_nodes()):
            u_data = rx_graph.get_node_data(u_idx)
            u = u_data.get("__networkx_node__", u_idx)
            
            for v_idx, data in rx_graph.edge_list(u_idx):
                v_data = rx_graph.get_node_data(v_idx)
                v = v_data.get("__networkx_node__", v_idx)
                if not nx_graph.has_edge(u, v):  # Avoid duplicates
                    nx_graph.add_edge(u, v, **data)
    
    return nx_graph

class BikePathEnv(gym.Env):
    """
    Custom Environment for optimizing bike path additions to a city network.
    
    This environment uses real city street networks from OSMnx and allows an RL agent
    to suggest optimal bike path additions based on metrics like connectivity, safety,
    and route directness.
    """
    
    def __init__(self, city_name=None, bbox=None, budget=1000, edge_cost_factor=10):
        super(BikePathEnv, self).__init__()
        
        if city_name is not None:
            print(f"Initializing environment for {city_name}")
            # Load the city graph using city name
            self.original_graph = ox.graph_from_place(city_name, network_type='bike')
            self.city_name = city_name
            self.location_type = "city"
            
            # Create a working copy of the graph
            self.graph = self.original_graph.copy()
            
            # Get all roads that don't already have bike infrastructure
            self.walk_graph = ox.graph_from_place(city_name, network_type='walk')
        
        elif bbox is not None:
            north, south, east, west = bbox
            print(f"Initializing environment for bbox: {north}, {south}, {east}, {west}")
            # Load the graph using bounding box coordinates
            self.original_graph = ox.graph_from_bbox(north, south, east, west, network_type='bike')
            self.bbox = bbox
            self.city_name = f"Area_({north:.4f},{south:.4f},{east:.4f},{west:.4f})"
            self.location_type = "bbox"
            
            # Create a working copy of the graph
            self.graph = self.original_graph.copy()
            
            # Get all roads that don't already have bike infrastructure
            self.walk_graph = ox.graph_from_bbox(north, south, east, west, network_type='walk')
        
        else:
            raise ValueError("Either city_name or bbox must be provided")
        
        # Load the city graph - osmnx returns NetworkX graphs
        self.original_graph = ox.graph_from_place(city_name, network_type='bike')
        self.city_name = city_name
        
        # Create a working copy of the graph
        self.graph = self.original_graph.copy()
        
        # Get all roads that don't already have bike infrastructure
        self.walk_graph = ox.graph_from_place(city_name, network_type='walk')
        
        # Add better validation of graphs
        print(f"Original bike graph: {len(self.original_graph.nodes())} nodes, {len(self.original_graph.edges())} edges")
        print(f"Walk graph: {len(self.walk_graph.nodes())} nodes, {len(self.walk_graph.edges())} edges")
        
        # Verify graph attributes
        self._validate_graph_attributes()
        
        # Identify candidate edges for bike path addition
        self.candidate_edges = self._get_candidate_edges()
        print(f"Found {len(self.candidate_edges)} candidate edges for bike paths")
        
        # Convert NetworkX graphs to rustworkx for better performance using the built-in converter
        # Store both versions for compatibility with osmnx
        self.rx_original_graph, self.rx_original_node_map = nx_to_rx(self.original_graph)
        self.rx_graph, self.rx_node_map = nx_to_rx(self.graph)
        self.rx_walk_graph, self.rx_walk_node_map = nx_to_rx(self.walk_graph)
        
        # Budget and parameters
        self.initial_budget = budget
        self.budget = budget
        self.edge_cost_factor = edge_cost_factor
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(self.candidate_edges))
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Initialize state
        self.state = self._get_state()
        self.added_paths = []
        
        # Initialize episode tracking
        self.episode_count = 0
        self.episode_steps = 0
        self.episode_reward = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Set update frequency based on graph size
        graph_size = len(self.walk_graph.edges())
        if graph_size > 5000:
            self.update_frequency = 5  # Only update every 5 steps for very large graphs
        elif graph_size > 1000:
            self.update_frequency = 3  # Update every 3 steps for large graphs
        else:
            self.update_frequency = 1  # Update every step for small graphs (current behavior)
        
        # Print initial state
        #print("Initial state:", self.state)


    def _validate_graph_attributes(self):
        """Validate and fix graph attributes."""
        #print("Validating graph attributes...")
        
        # Check and fix node attributes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if 'x' not in node_data or 'y' not in node_data:
                print(f"Warning: Node {node} missing coordinates")
                if node in self.walk_graph.nodes():
                    self.graph.nodes[node].update(self.walk_graph.nodes[node])
        
        # Check and fix edge attributes
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if 'length' not in data:
                print(f"Warning: Edge ({u}, {v}) missing length attribute")
                # Calculate length using node coordinates
                try:
                    u_coords = (self.graph.nodes[u]['y'], self.graph.nodes[u]['x'])
                    v_coords = (self.graph.nodes[v]['y'], self.graph.nodes[v]['x'])
                    data['length'] = ox.distance.great_circle(u_coords, v_coords)
                except Exception as e:
                    print(f"Error calculating length for edge ({u}, {v}): {e}")
        
    def _get_candidate_edges(self):
        """Identify roads that don't have bike infrastructure with main road prioritization."""
        candidates = []
        
        # Get existing bike edges
        bike_edges = set((u, v) for u, v, _ in self.original_graph.edges(keys=True))
        validation_errors = 0
        
        # Store nodes that already have bike paths for continuity check
        self.bike_nodes = set()
        for u, v, _ in self.original_graph.edges(keys=True):
            self.bike_nodes.add(u)
            self.bike_nodes.add(v)
        
        # Iterate through walkable edges
        for u, v, data in self.walk_graph.edges(data=True):
            try:
                if (u, v) not in bike_edges and (v, u) not in bike_edges:
                    # Validate road type
                    road_type = data.get('highway', '')
                    if isinstance(road_type, list):
                        road_type = road_type[0] if road_type else ''
                    
                    # Define road type priorities (higher = more priority for bike paths)
                    road_priorities = {
                        'primary': 5,      # Main arterial roads - highest priority
                        'secondary': 4,    # Important urban roads
                        'tertiary': 3,     # Connecting local roads
                        'residential': 2,  # Residential streets
                        'unclassified': 1, # Minor roads
                        #'living_street': 1 # Very low traffic residential
                    }
                    
                    suitable_types = set(road_priorities.keys())
                    
                    if road_type in suitable_types:
                        # Validate node coordinates
                        u_coords = (self.walk_graph.nodes[u].get('y'), self.walk_graph.nodes[u].get('x'))
                        v_coords = (self.walk_graph.nodes[v].get('y'), self.walk_graph.nodes[v].get('x'))
                        
                        if all(u_coords) and all(v_coords):
                            # Calculate or validate length
                            if 'length' not in data:
                                data['length'] = ox.distance.great_circle(u_coords, v_coords)
                            
                            # Additional validation
                            if data['length'] > 100 and data['length'] < 1000:
                                # Add road priority to data
                                data['road_priority'] = road_priorities.get(road_type, 0)
                                
                                # Check if this edge would connect to existing bike infrastructure
                                data['connects_to_bike_path'] = (u in self.bike_nodes) or (v in self.bike_nodes)
                                
                                candidates.append((u, v, data))
                        else:
                            validation_errors += 1
                            if validation_errors < 10:  # Limit error messages
                                print(f"Warning: Missing coordinates for edge ({u}, {v})")
            except Exception as e:
                validation_errors += 1
                if validation_errors < 10:
                    print(f"Error processing edge ({u}, {v}): {e}")
        
        if validation_errors > 0:
            print(f"Encountered {validation_errors} validation errors during candidate edge search")
        
        # Sort candidates by road priority (higher first) and connectivity to existing paths
        candidates = sorted(candidates, key=lambda x: (x[2]['road_priority'], x[2]['connects_to_bike_path']), reverse=True)
        
        return candidates
    
    def _calculate_fragmentation(self):
        """Calculate a fragmentation score for the current bike network.
        Lower values mean less fragmentation (better connectivity).
        """
        try:
            # Convert to undirected graph for component analysis
            import networkx as nx
            undirected = nx.Graph()
            
            # Add all nodes and edges from the bike graph
            for u, v, data in self.graph.edges(data=True):
                if 'bike_lane' in data and data['bike_lane'] == 'yes':
                    undirected.add_edge(u, v)
            
            # If no bike lanes, return maximum fragmentation
            if len(undirected.edges()) == 0:
                return 1.0
                
            # Find all connected components
            components = list(nx.connected_components(undirected))
            
            if not components:
                return 1.0
                
            # More components = more fragmentation
            num_components = len(components)
            
            # Largest component size as fraction of total
            largest_component = max(components, key=len)
            largest_fraction = len(largest_component) / undirected.number_of_nodes()
            
            # Components of size 1 or 2 are considered isolated fragments
            small_components = sum(1 for comp in components if len(comp) <= 2)
            isolation_factor = small_components / max(1, num_components)
            
            # Calculate final fragmentation score (weighted average)
            # - More components = higher fragmentation
            # - Smaller main component = higher fragmentation
            # - More isolated segments = higher fragmentation
            fragmentation = (
                0.3 * min(1.0, (num_components - 1) / 10) +  # Scale component count
                0.4 * (1.0 - largest_fraction) +            # Inverse of largest component size
                0.3 * isolation_factor                      # Penalty for isolated fragments
            )
            
            return max(0.0, min(1.0, fragmentation))
            
        except Exception as e:
            print(f"Error calculating fragmentation: {e}")
            return 1.0  # Maximum fragmentation on error
        
    def _calculate_connectivity(self):
        """Calculate connectivity metrics using rustworkx's built-in parallel functionality with optimization."""
        try:
            # Verify graph has nodes and edges
            if len(self.graph.nodes()) == 0 or len(self.graph.edges()) == 0:
                print("Warning: Graph has no nodes or edges")
                return 0.0, 0.0
            
            current_edge_count = len(self.graph.edges())
            if hasattr(self, '_last_connectivity_edges') and \
            self._last_connectivity_edges == current_edge_count and \
            hasattr(self, '_cached_connectivity') and \
            hasattr(self, '_cached_path_efficiency'):
                return self._cached_connectivity, self._cached_path_efficiency
                    
            # Convert multigraph to simple graph
            import networkx as nx  # Import here for compatibility
            simple_graph = nx.Graph()
            for u, v, data in self.graph.edges(data=True):
                if simple_graph.has_edge(u, v):
                    existing_length = simple_graph[u][v]['length']
                    if data.get('length', float('inf')) < existing_length:
                        simple_graph[u][v]['length'] = data['length']
                else:
                    simple_graph.add_edge(u, v, length=data.get('length', 1.0))
            
            # Convert to rustworkx for better performance
            rx_simple_graph, node_map = nx_to_rx(simple_graph)
            
            # Get largest connected component using rustworkx
            rx_components = rx.connected_components(rx_simple_graph)
            
            if not rx_components:
                return 0.0, 0.0
                
            largest_cc = max(rx_components, key=len)
            
            # Create subgraph of largest component
            rx_subgraph = rx_simple_graph.subgraph(list(largest_cc), preserve_attrs=True)
            
            # OPTIMIZATION 3: Approximated clustering coefficient for large graphs
            graph_size = rx_subgraph.num_nodes()
            if graph_size > 10000:
                # Use sampling approach for extremely large graphs
                clustering_sample_size = min(40, graph_size)  # Reduced from 100
                sampled_nodes = random.sample(range(graph_size), clustering_sample_size)
                
                clustering_values = []
                for node in sampled_nodes:
                    try:
                        # Calculate local clustering coefficient for sample nodes
                        local_clustering = rx.local_clustering_coefficient(rx_subgraph, node)
                        if local_clustering is not None:
                            clustering_values.append(local_clustering)
                    except:
                        continue
                
                # Average the sampled values
                clustering = np.mean(clustering_values) if clustering_values else 0.0
            elif graph_size > 5000:
                # Use sampling approach for very large graphs
                clustering_sample_size = min(60, graph_size)  # Reduced from 100
                sampled_nodes = random.sample(range(graph_size), clustering_sample_size)
                
                clustering_values = []
                for node in sampled_nodes:
                    try:
                        # Calculate local clustering coefficient for sample nodes
                        local_clustering = rx.local_clustering_coefficient(rx_subgraph, node)
                        if local_clustering is not None:
                            clustering_values.append(local_clustering)
                    except:
                        continue
                
                # Average the sampled values
                clustering = np.mean(clustering_values) if clustering_values else 0.0
            else:
                # Use global transitivity for smaller graphs (original code)
                clustering = rx.transitivity(rx_subgraph)
                if clustering is None:
                    clustering = 0.0
                
            # OPTIMIZATION 2: Calculate average path length with MUCH MORE aggressive sampling
            path_lengths = []

            # Much more aggressive sampling strategy for large graphs
            if graph_size > 10000:
                # Extremely aggressive sampling for extremely large graphs
                sample_size = max(1, int(math.log(graph_size)))
                sampled_nodes = random.sample(range(graph_size), min(sample_size, graph_size))
            elif graph_size > 5000:
                # Very aggressive sampling for very large graphs
                sample_size = max(2, int(math.log(graph_size) * 1.2))
                sampled_nodes = random.sample(range(graph_size), min(sample_size, graph_size))
            elif graph_size > 1000:
                # More aggressive sampling for large graphs
                sample_size = max(3, int(math.log(graph_size) * 1.5))
                sampled_nodes = random.sample(range(graph_size), min(sample_size, graph_size))
            elif graph_size > 100:
                # Square root sampling for medium graphs
                sample_size = max(8, int(math.sqrt(graph_size)))
                sampled_nodes = random.sample(range(graph_size), min(sample_size, graph_size))
            else:
                # Use all nodes for small graphs
                sampled_nodes = list(range(graph_size))

            # Define weight function for rustworkx
            def weight_fn(edge_data):
                return edge_data.get('length', 1.0)

            # Limit target nodes for each source to reduce calculations
            for source in sampled_nodes:
                try:
                    # Calculate distances to all nodes from this source in one call
                    distances = rx.dijkstra_shortest_path_lengths(rx_subgraph, source, weight_fn)
                    
                    # Limit target nodes based on graph size
                    target_count = 2 if graph_size > 5000 else 3
                    targets = random.sample(sampled_nodes, min(target_count, len(sampled_nodes)))
                    for target in targets:
                        if target != source and target in distances and distances[target] > 0:
                            path_lengths.append(distances[target])
                except Exception as e:
                    # Don't exit on error, just continue
                    continue
            
            # Calculate metrics
            if path_lengths:
                avg_path_length = np.mean(path_lengths)
            else:
                print("Warning: No valid path lengths calculated")
                avg_path_length = float('inf')
            
            # Normalize metrics
            normalized_clustering = min(max(clustering, 0.0), 1.0)
            if avg_path_length == float('inf'):
                normalized_path = 0.0
            else:
                normalized_path = 1.0 / (1.0 + avg_path_length/1000)
            
            self._cached_connectivity = normalized_clustering
            self._cached_path_efficiency = normalized_path
            self._last_connectivity_edges = current_edge_count

            return normalized_clustering, normalized_path
                
        except Exception as e:
            print(f"Connectivity calculation failed: {e}")
            return 0.0, 0.0
    
    def _update_path_efficiency_incremental(self, u, v):
        """Incrementally update path efficiency when a new edge connects to the network."""
        # Get current path efficiency
        current_efficiency = self.state[1]
        
        # Define weight function for rustworkx
        def weight_fn(edge_data):
            return edge_data.get('length', 1.0)
        
        # Convert node IDs to rustworkx indices
        if u in self.rx_node_map and v in self.rx_node_map:
            u_rx = self.rx_node_map[u]
            v_rx = self.rx_node_map[v]
            
            # Calculate a sample of new paths made possible by this edge
            path_improvements = []
            
            # Sample at most 5 nodes from each side of the new edge
            u_neighbors = list(self.rx_graph.neighbors(u_rx))[:5]
            v_neighbors = list(self.rx_graph.neighbors(v_rx))[:5]
            
            for src in u_neighbors:
                for dst in v_neighbors:
                    if src != dst:
                        try:
                            # Check if a shorter path is now possible
                            old_dist = rx.dijkstra_shortest_path_length(
                                self.rx_graph, src, dst, weight_fn)
                            
                            # Approximate new distance through the new edge
                            u_to_src_dist = rx.dijkstra_shortest_path_length(
                                self.rx_graph, u_rx, src, weight_fn)
                            v_to_dst_dist = rx.dijkstra_shortest_path_length(
                                self.rx_graph, v_rx, dst, weight_fn)
                            
                            # Get edge weight between u and v
                            edge_weight = weight_fn(self.rx_graph.get_edge_data(u_rx, v_rx))
                            
                            new_dist = u_to_src_dist + edge_weight + v_to_dst_dist
                            
                            if new_dist < old_dist:
                                # Path improved, calculate the improvement factor
                                improvement = (old_dist - new_dist) / old_dist
                                path_improvements.append(improvement)
                        except:
                            continue
            
            # If we found improvements, update path efficiency
            if path_improvements:
                avg_improvement = sum(path_improvements) / len(path_improvements)
                # Apply a weighted update to current efficiency
                # Use a small factor (0.1) to avoid overestimating the impact
                updated_efficiency = current_efficiency * (1 + 0.1 * avg_improvement)
                return min(1.0, updated_efficiency)
        
        # If we can't calculate incremental update, return current value
        return current_efficiency

    def _calculate_population_served(self):
        """Calculate population served with improved metrics using rustworkx."""
        try:
            if len(self.graph.nodes()) == 0:
                return 0.0
            
            # Calculate based on both network coverage and connectivity
            bike_nodes = len(self.graph.nodes())
            all_nodes = len(self.walk_graph.nodes())
            
            # Basic coverage ratio
            coverage_ratio = bike_nodes / max(1, all_nodes)
            
            # Convert NetworkX graph to undirected rustworkx graph
            import networkx as nx  # Import here for compatibility
            undirected_graph = self.graph.to_undirected()
            rx_undirected = rx.networkx_converter(undirected_graph, keep_attributes=True)
            
            # Get connected components using rustworkx
            rx_components = rx.connected_components(rx_undirected)
            
            if not rx_components:
                return 0.0
                
            # Calculate component size ratio
            largest_component = max(rx_components, key=len)
            component_ratio = len(largest_component) / max(1, bike_nodes)
            
            # Combine metrics with weights
            population_score = 0.7 * coverage_ratio + 0.3 * component_ratio
            
            return min(1.0, population_score)
            
        except Exception as e:
            print(f"Population served calculation failed: {e}")
            return 0.0
        
    def _get_state(self):
        """Get current state with improved metrics and validation."""
        try:
            # Calculate connectivity metrics
            connectivity, path_efficiency = self._calculate_connectivity()
            
            # Calculate population served
            pop_served = self._calculate_population_served()
            
            # Calculate normalized budget
            norm_budget = self.budget / self.initial_budget
            
            # Calculate fragmentation score (lower is better)
            fragmentation_score = self._calculate_fragmentation()
            
            # Validate metrics
            state = np.array([
                max(0.0, min(1.0, connectivity)),
                max(0.0, min(1.0, path_efficiency)),
                max(0.0, min(1.0, pop_served)),
                max(0.0, min(1.0, norm_budget))
            ], dtype=np.float32)

            # Store fragmentation as additional info, not part of the state vector
            self.current_fragmentation = fragmentation_score
            
            return state
                
        except Exception as e:
            print(f"Error calculating state: {e}")
            return np.zeros(4, dtype=np.float32)
        
    
    def step(self, action):
        """Take an action with improved anti-fragmentation measures and incremental updates."""
        self.episode_steps += 1
        self.total_steps += 1
        
        # Validate action
        if action >= len(self.candidate_edges):
            return self.state, -10, True, False, {}
        
        # Get selected edge
        u, v, data = self.candidate_edges[action]
        
        # Store old state before modification
        old_state = self.state.copy()
        old_fragmentation = getattr(self, 'current_fragmentation', 1.0)
        
        # Validate edge cost
        cost = data.get('length', 0) * self.edge_cost_factor
        if cost <= 0:
            return self.state, -5, False, False, {}
        
        # Check budget
        if cost > self.budget:
            self.candidate_edges.pop(action)
            
            # Check if we're done
            min_edge_cost = min([d['length'] * self.edge_cost_factor for _, _, d in self.candidate_edges]) if self.candidate_edges else float('inf')
            done = (self.budget < min_edge_cost) or len(self.candidate_edges) == 0
            
            return self.state, -1, done, False, {}
        
        # Check if this would create an isolated segment (not connected to existing network)
        would_create_isolated = False
        connects_to_bike_path = data.get('connects_to_bike_path', False)
        
        if not connects_to_bike_path and len(self.added_paths) > 0:
            # This would create a disconnected segment
            would_create_isolated = True
        
        # Update budget
        self.budget -= cost
        
        # Add edge to graph with proper attribute handling
        if not self.graph.has_edge(u, v):
            # Ensure nodes exist with proper attributes
            if u not in self.graph:
                self.graph.add_node(u, **self.walk_graph.nodes[u])
            if v not in self.graph:
                self.graph.add_node(v, **self.walk_graph.nodes[v])
                
            # Add edge with validated attributes
            edge_data = data.copy()
            edge_data['bike_lane'] = 'yes'  # Mark as bike path
            if 'length' not in edge_data:
                # Calculate length if missing
                u_coords = (self.graph.nodes[u]['y'], self.graph.nodes[u]['x'])
                v_coords = (self.graph.nodes[v]['y'], self.graph.nodes[v]['x'])
                edge_data['length'] = ox.distance.great_circle(u_coords, v_coords)
            
            self.graph.add_edge(u, v, **edge_data)
            
            # Also update the rustworkx graph
            if u in self.rx_node_map and v in self.rx_node_map:
                u_rx = self.rx_node_map[u]
                v_rx = self.rx_node_map[v]
                self.rx_graph.add_edge(u_rx, v_rx, edge_data)
            else:
                # Convert again if node maps need updating
                self.rx_graph, self.rx_node_map = nx_to_rx(self.graph)
                
            # Add to the set of nodes with bike paths (for continuity tracking)
            self.bike_nodes.add(u)
            self.bike_nodes.add(v)
            
            self.added_paths.append((u, v, edge_data['length']))
        
        # OPTIMIZATION 1: Use incremental state updates instead of full recalculation
        if connects_to_bike_path:
            # Edge connects to existing network - update path efficiency incrementally
            path_efficiency = self._update_path_efficiency_incremental(u, v)
            self.state[1] = path_efficiency
            
            # Update normalized budget directly
            self.state[3] = self.budget / self.initial_budget
            
            # We still need to update fragmentation score for proper rewards
            old_fragmentation = self.current_fragmentation
            self.current_fragmentation = self._calculate_fragmentation()
        else:
            # Edge creates a new component - need full state recalculation
            old_state = self.state
            self.state = self._get_state()
        
        # Calculate reward with more weight on connectivity improvements
        state_improvement = [
            (new - old) * weight for new, old, weight in 
            zip(self.state[:3], old_state[:3], [0.9, 0.2, 0.3])  # connectivity, efficiency, population
        ]
        reward = sum(state_improvement) * 100
        
        # Add bonus for main roads (road priority bonus)
        road_priority = data.get('road_priority', 0)
        road_priority_bonus = road_priority * 5  # Scale the bonus based on road priority
        reward += road_priority_bonus
        
        # Add continuity bonus if this edge connects to existing bike paths
        # Much stronger now to discourage fragmentation
        if connects_to_bike_path:
            continuity_bonus = 50  # Significantly higher reward for creating continuous paths
            reward += continuity_bonus
        
        # Add fragmentation improvement bonus (or penalty)
        fragmentation_improvement = old_fragmentation - self.current_fragmentation
        fragmentation_reward = fragmentation_improvement * 200  # High weight for reducing fragmentation
        reward += fragmentation_reward
        
        # Strong penalty for creating isolated segments
        if would_create_isolated:
            isolation_penalty = -100
            reward += isolation_penalty
        
        # Add reward for efficient budget use
        budget_efficiency = cost / (data['length'] + 1)  # Lower cost per meter is better
        reward += max(0, (1 - budget_efficiency/10) * 10)  # Scale and cap budget efficiency reward
        
        self.episode_reward += reward
        
        # Update candidate edges to reflect new connectivity information
        for i, (cu, cv, cdata) in enumerate(self.candidate_edges):
            # Update connectivity status for remaining candidate edges
            self.candidate_edges[i][2]['connects_to_bike_path'] = (cu in self.bike_nodes) or (cv in self.bike_nodes)
        
        # Re-sort candidate edges by priority and connectivity after each step
        # Now with much stronger emphasis on connectivity to existing network
        self.candidate_edges = sorted(self.candidate_edges, 
                                    key=lambda x: (
                                        # Connected segments get highest priority by far
                                        10 if x[2]['connects_to_bike_path'] else 0,
                                        # Road type is secondary priority
                                        x[2]['road_priority']
                                    ), 
                                    reverse=True)
        
        # Remove used edge from the list of candidates
        for i, (cu, cv, _) in enumerate(self.candidate_edges):
            if cu == u and cv == v:
                self.candidate_edges.pop(i)
                break
        
        min_edge_cost = min([d['length'] * self.edge_cost_factor for _, _, d in self.candidate_edges]) if self.candidate_edges else float('inf')
        done = (self.budget < min_edge_cost) or len(self.candidate_edges) == 0
        
        # Create detailed info dict
        info = {
            "budget_remaining": self.budget,
            "budget_used_pct": 1.0 - (self.budget / self.initial_budget),
            "paths_added": len(self.added_paths),
            "connectivity": self.state[0],
            "path_efficiency": self.state[1],
            "population_served": self.state[2],
            "fragmentation": self.current_fragmentation,
            "action_index": action,
            "edge_cost": cost,
            "edge_length": data['length'],
            "road_priority": road_priority,
            "connects_to_bike_path": connects_to_bike_path,
            "would_create_isolated": would_create_isolated,
            "actions_remaining": len(self.candidate_edges)
        }

        return self.state, reward, done, False, info

    
    def reset(self, seed=None, options=None):
        """Reset environment with improved initialization."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Store previous episode data before incrementing episode count
        if self.episode_count > 0:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_steps)
        
        self.episode_count += 1
        
        # Print episode summary
        if self.episode_count > 1 and self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        # Reset graph and validate
        self.graph = self.original_graph.copy()
        self._validate_graph_attributes()
        
        # Reset rustworkx graph
        self.rx_graph, self.rx_node_map = nx_to_rx(self.graph)
        
        # Initialize the set of nodes with bike paths
        self.bike_nodes = set()
        for u, v, _ in self.original_graph.edges(keys=True):
            self.bike_nodes.add(u)
            self.bike_nodes.add(v)
        
        # Reset other variables
        self.budget = self.initial_budget
        self.candidate_edges = self._get_candidate_edges()
        self.added_paths = []
        self.episode_steps = 0
        self.episode_reward = 0
        
        # Get initial state
        self.state = self._get_state()
        
        return self.state, {}
    
    def render(self, mode='human', eval_num=None):
        """Render with improved visualization, timesteps, and evaluation number information."""
        try:
            fig, ax = plt.subplots(figsize=(15, 12))
            
            # Plot base network
            if len(self.original_graph.edges()) > 0:
                ox.plot_graph(self.original_graph, ax=ax, node_size=0, 
                            edge_color='lightblue', edge_linewidth=1, 
                            edge_alpha=0.7, show=False)
            else:
                print("Warning: Original graph has no edges to plot")
            
            # Plot added paths
            if self.added_paths:
                edges_gdf = self.get_edge_gdf()
                if edges_gdf is not None and not edges_gdf.empty:
                    edges_gdf.plot(ax=ax, color='red', linewidth=3, alpha=1)
                    
                    # Add stats annotation
                    total_length = edges_gdf['length'].sum()
                    stats_text = (
                        f"Added Paths: {len(self.added_paths)}\n"
                        f"Total Length: {total_length:.1f}m\n"
                        f"Budget Used: {(self.initial_budget - self.budget):.1f}"
                    )
                    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                            bbox=dict(facecolor='white', alpha=0.8),
                            verticalalignment='top')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='lightblue', lw=1, alpha=0.7, label='Existing Network'),
                Line2D([0], [0], color='red', lw=3, alpha=1, label='Added Bike Paths')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Add metrics to title with timesteps and evaluation information
            conn, path_eff, pop, budget_remain = self.state
            title = f"Bike Network Plan - {self.city_name} (Timestep: {self.total_steps}"
            
            # Add evaluation number if provided
            if eval_num is not None:
                title += f", Evaluation: {eval_num}"
            
            title += f")\nConnectivity: {conn:.3f} | Path Efficiency: {path_eff:.3f} | "
            title += f"Population Served: {pop:.3f}\n"
            title += f"Budget Remaining: {self.budget:.0f}/{self.initial_budget:.0f} "
            title += f"({(1-budget_remain)*100:.1f}% used)"
            
            plt.title(title)
            
            # Save figure with timesteps and evaluation number in the filename
            # Use the run-specific directory if available
            if RUN_FIGURES_DIR is not None:
                save_dir = RUN_FIGURES_DIR
            else:
                save_dir = "./bike_path_figures"
                os.makedirs(save_dir, exist_ok=True)
                
            # Build filename
            filename = f"{self.city_name.replace(', ', '_').replace(' ', '_')}_step{self.total_steps}"
            if eval_num is not None:
                filename += f"_eval{eval_num}"
            filename += ".png"
            
            fig_path = os.path.join(save_dir, filename)
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_path}")
            
            plt.show()
            return fig
                
        except Exception as e:
            print(f"Error in render: {e}")
            return None
        
    def get_edge_gdf(self):
        """Convert added paths to GeoDataFrame with improved error handling."""
        try:
            edges = []
            for u, v, length in self.added_paths:
                try:
                    if self.graph.has_node(u) and self.graph.has_node(v):
                        u_data = self.graph.nodes[u]
                        v_data = self.graph.nodes[v]
                        
                        if all(k in u_data for k in ['x', 'y']) and all(k in v_data for k in ['x', 'y']):
                            geom = LineString([
                                (u_data['x'], u_data['y']),
                                (v_data['x'], v_data['y'])
                            ])
                            
                            edges.append({
                                'geometry': geom,
                                'length': length,
                                'start_node': u,
                                'end_node': v
                            })
                        else:
                            print(f"Warning: Missing coordinates for edge ({u}, {v})")
                except Exception as e:
                    print(f"Error processing edge ({u}, {v}): {e}")
                    continue
            
            if edges:
                gdf = gpd.GeoDataFrame(edges, crs="EPSG:4326")
                return gdf
            else:
                print("No valid edges to convert to GeoDataFrame")
                return None
                
        except Exception as e:
            print(f"Error creating GeoDataFrame: {e}")
            return None


class TrainingProgressCallback(BaseCallback):
    """
    Progress callback showing a progress bar during training,
    accounting for multiple parallel environments, and tracking rewards over time.
    """
    def __init__(self, total_timesteps):
        super(TrainingProgressCallback, self).__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.training_start = None
        self.episode_rewards = []
        self.episode_timesteps = []  # Store timesteps for each episode reward
        self.last_timesteps = 0
        
    def _on_training_start(self):
        self.training_start = time.time()
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")
        
    def _on_step(self):
        if self.pbar:
            # Get the actual number of environments used
            n_envs = self.model.get_env().num_envs
            
            # Calculate steps done since last update (taking into account parallel envs)
            steps_done = self.num_timesteps - self.last_timesteps
            self.last_timesteps = self.num_timesteps
            
            # Update progress bar
            self.pbar.update(steps_done)
            
            # Update reward description if available
            if len(self.locals["infos"]) > 0:
                for info in self.locals["infos"]:
                    if "episode" in info:
                        reward = info["episode"]["r"]
                        self.episode_rewards.append(reward)
                        self.episode_timesteps.append(self.num_timesteps)  # Store current timestep
                        avg_reward = np.mean(self.episode_rewards[-100:])
                        self.pbar.set_description(
                            f"Training ({n_envs} envs) | Avg reward: {avg_reward:.2f}"
                        )
        
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
            
    def get_reward_data(self):
        """Return the reward and timestep data for plotting."""
        return self.episode_timesteps, self.episode_rewards

class EvaluationTracker:
    """
    Tracks and summarizes evaluation metrics across multiple episodes.
    """
    def __init__(self, num_evaluations):
        self.num_evaluations = num_evaluations
        self.pbar = None
        self.start_time = None
        
        # Metrics storage
        self.rewards = []
        self.num_paths = []
        self.connectivity = []
        self.path_efficiency = []
        self.population = []
        self.budget_used = []
        
        # Best solution tracking
        self.best_reward = -float('inf')
        self.best_solution = None
    
    def start_evaluation(self):
        """Initialize evaluation progress tracking"""
        self.start_time = time.time()
        self.pbar = tqdm(total=self.num_evaluations, desc="Evaluation Progress")
    
    def update_episode(self, episode_metrics):
        """Update metrics after each evaluation episode"""
        # Store metrics
        self.rewards.append(episode_metrics['reward'])
        self.num_paths.append(episode_metrics['num_paths'])
        self.connectivity.append(episode_metrics['connectivity'])
        self.path_efficiency.append(episode_metrics['path_efficiency'])
        self.population.append(episode_metrics['population'])
        self.budget_used.append(episode_metrics['budget_used'])
        
        # Track best solution
        if episode_metrics['reward'] > self.best_reward:
            self.best_reward = episode_metrics['reward']
            self.best_solution = episode_metrics
        
        # Update progress bar
        if self.pbar:
            avg_reward = np.mean(self.rewards)
            self.pbar.set_description(
                f"Evaluation Progress | Avg Reward: {avg_reward:.2f}"
            )
            self.pbar.update(1)
    
    def get_summary(self):
        """Generate evaluation summary statistics"""
        total_time = time.time() - self.start_time
        
        summary = {
            'time_elapsed': str(timedelta(seconds=int(total_time))),
            'episodes': len(self.rewards),
            'avg_reward': np.mean(self.rewards),
            'std_reward': np.std(self.rewards),
            'avg_paths': np.mean(self.num_paths),
            'avg_connectivity': np.mean(self.connectivity),
            'avg_path_efficiency': np.mean(self.path_efficiency),
            'avg_population': np.mean(self.population),
            'avg_budget_used': np.mean(self.budget_used),
            'best_solution': self.best_solution
        }
        
        return summary
    
    def print_summary(self):
        """Print formatted evaluation summary"""
        summary = self.get_summary()
        
        print("\nEvaluation Summary:")
        print("-" * 50)
        print(f"Total evaluation time: {summary['time_elapsed']}")
        print(f"Episodes completed: {summary['episodes']}")
        print(f"Average reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Average paths added: {summary['avg_paths']:.1f}")
        print(f"Average metrics:")
        print(f"  - Connectivity: {summary['avg_connectivity']:.4f}")
        print(f"  - Path Efficiency: {summary['avg_path_efficiency']:.4f}")
        print(f"  - Population Coverage: {summary['avg_population']:.4f}")
        print(f"  - Budget Utilization: {summary['avg_budget_used']*100:.1f}%")
        print("\nBest Solution:")
        print(f"  - Reward: {self.best_reward:.2f}")
        print(f"  - Paths: {self.best_solution['num_paths']}")
        print(f"  - Connectivity: {self.best_solution['connectivity']:.4f}")
        print(f"  - Population Coverage: {self.best_solution['population']:.4f}")
        print("-" * 50)
    
    def close(self):
        """Clean up progress bar"""
        if self.pbar:
            self.pbar.close()

def make_env(city_name=None, bbox=None, budget=100000, rank=0):
    """
    Function that creates an environment with a specific seed for process isolation
    """
    def _init():
        env = BikePathEnv(city_name=city_name, bbox=bbox, budget=budget)
        env.reset(seed=rank)  # Different seed for each environment
        return env
    return _init

def train_rl_model(city_name=None, bbox=None, budget=100000, total_timesteps=10240, device="cpu", n_envs=2, plot_rewards=True):
    """Train RL model with parallel environments for improved performance."""
    # Create environments in separate processes
    env_fns = [make_env(city_name=city_name, bbox=bbox, budget=budget, rank=i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    
    # Print environment info
    print(f"Training with {n_envs} parallel environments")
    
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device=device
    )
    
    # Setup callbacks
    progress_callback = TrainingProgressCallback(total_timesteps)
    
    # Create checkpoint directory within the run directory
    checkpoint_dir = os.path.join(RUN_FIGURES_DIR, "checkpoints") if RUN_FIGURES_DIR else "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1000),
        save_path=checkpoint_dir,
        name_prefix="bike_model",
        verbose=0
    )
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, checkpoint_callback]
    )
    
    # Plot reward history using the run directory if available
    if plot_rewards:
        timesteps, rewards = progress_callback.get_reward_data()
        plot_reward_vs_timestep(
            rewards=rewards, 
            timesteps=timesteps, 
            city_name=city_name,
            smoothing=10,
            save_path=os.path.join(RUN_FIGURES_DIR, "training_rewards.png") if RUN_FIGURES_DIR else None
        )
    
    # Close environment processes when done
    env.close()
    
    return model


def evaluate_and_visualize(model, city_name=None, bbox=None, budget=10000, num_evaluations=5):
    """Evaluate model with improved progress tracking and evaluation numbers."""
    # Initialize evaluation tracker
    tracker = EvaluationTracker(num_evaluations)
    tracker.start_evaluation()
    
    try:
        for eval_num in range(1, num_evaluations + 1):
            # Create fresh environment
            env = BikePathEnv(city_name=city_name, bbox=bbox, budget=budget)
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            
            while not done and not truncated:
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                obs, reward, done, truncated, info = env.step(int(action[0]))
                total_reward += reward
            
            # Collect episode metrics
            conn, path_eff, pop, remaining_budget = env.state
            metrics = {
                'reward': total_reward,
                'num_paths': len(env.added_paths),
                'connectivity': conn,
                'path_efficiency': path_eff,
                'population': pop,
                'budget_used': 1 - remaining_budget,
                'paths': env.added_paths,
                'env': env,
                'eval_num': eval_num  # Store evaluation number
            }
            
            tracker.update_episode(metrics)
    
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise
    finally:
        tracker.close()
    
    # Print evaluation summary
    tracker.print_summary()
    
    # Visualize best solution with evaluation number
    if tracker.best_solution:
        best_env = tracker.best_solution['env']
        best_eval_num = tracker.best_solution.get('eval_num', None)
        best_env.render(mode='human', eval_num=best_eval_num)
        
        '''
        # Export results if available
        try:
            gdf = best_env.get_edge_gdf()
            if gdf is not None:
                safe_name = city_name.replace(', ', '_').replace(' ', '_')
                eval_tag = f"_eval{best_eval_num}" if best_eval_num else ""
                
                # Use run directory if available
                if RUN_FIGURES_DIR:
                    export_path = os.path.join(RUN_FIGURES_DIR, f"suggested_bike_paths{eval_tag}.geojson")
                else:
                    export_path = f"suggested_bike_paths_{safe_name}{eval_tag}.geojson"
                
                gdf.to_file(export_path, driver='GeoJSON')
        except Exception as e:
            print(f"Error exporting results: {e}")
        '''
        
    # Plot evaluation results
    plot_evaluation_results(tracker, city_name)
    
    return tracker.best_solution['paths'], tracker.best_solution['env'].state if tracker.best_solution else (None, None)

def plot_reward_vs_timestep(rewards, timesteps=None, city_name=None, smoothing=5, 
                           save_path=None, show=True):
    """
    Plot reward vs timestep or episode from training or evaluation.
    
    Parameters:
    -----------
    rewards : list
        List of episode rewards.
    timesteps : list, optional
        List of timesteps for each episode. If None, will use episode numbers.
    city_name : str, optional
        City name for the figure title.
    smoothing : int, optional
        Smoothing window size. Set to 0 for no smoothing.
    save_path : str, optional
        If provided, save the figure to this path.
    show : bool, optional
        Whether to display the plot. Default is True.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime
    
    if not rewards:
        print("No reward data available for plotting.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine x-axis: timesteps or episode numbers
    if timesteps and len(timesteps) == len(rewards):
        x_values = timesteps
        x_label = "Timesteps"
    else:
        x_values = np.arange(1, len(rewards) + 1)
        x_label = "Episode"
    
    # Plot raw rewards
    ax.scatter(x_values, rewards, alpha=0.4, color='blue', label='Rewards')
    
    # Add smoothed curve if requested
    if smoothing > 0 and len(rewards) > smoothing * 2:
        try:
            # Try to use scipy for better smoothing if available
            from scipy.ndimage import gaussian_filter1d
            y_smooth = gaussian_filter1d(rewards, sigma=smoothing)
            ax.plot(x_values, y_smooth, linewidth=2, color='red', 
                    label=f'Smoothed (σ={smoothing})')
        except ImportError:
            # Fall back to moving average if scipy not available
            window_size = min(len(rewards), max(3, smoothing * 2))
            smoothed = []
            for i in range(len(rewards)):
                start = max(0, i - window_size // 2)
                end = min(len(rewards), i + window_size // 2 + 1)
                window = rewards[start:end]
                smoothed.append(sum(window) / len(window))
            ax.plot(x_values, smoothed, linewidth=2, color='red', 
                    label=f'Moving Avg (window={window_size})')
    
    # Add mean line
    mean_reward = np.mean(rewards)
    ax.axhline(y=mean_reward, color='green', linestyle='--', 
               label=f'Mean: {mean_reward:.2f}')
    
    # Set labels and title
    title = 'Reward History'
    if city_name:
        title += f' - {city_name}'
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
   
    return fig

def plot_evaluation_results(evaluation_tracker, city_name=None):
    """
    Plot comprehensive evaluation results from an EvaluationTracker instance.
    
    Parameters:
    -----------
    evaluation_tracker : EvaluationTracker
        The evaluation tracker with result metrics
    city_name : str, optional
        City name for plot titles
    
    Returns:
    --------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime
    
    if not hasattr(evaluation_tracker, 'rewards') or not evaluation_tracker.rewards:
        print("No evaluation data available for plotting.")
        return
    
    # Determine save directory - use run directory if available
    if RUN_FIGURES_DIR:
        save_dir = RUN_FIGURES_DIR
    else:
        save_dir = "./bike_path_figures"
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot rewards from evaluation episodes
    plot_reward_vs_timestep(
        rewards=evaluation_tracker.rewards,
        city_name=f"{city_name} (Evaluation)" if city_name else "Evaluation",
        smoothing=0,  # No smoothing for evaluation as we typically have fewer episodes
        save_path=os.path.join(save_dir, "evaluation_rewards.png"),
        show=False  # Don't show yet, we'll create a more comprehensive plot
    )
    
    # 2. Create comprehensive metrics plot
    metrics = {
        'Reward': evaluation_tracker.rewards,
        'Number of Paths': evaluation_tracker.num_paths,
        'Connectivity': evaluation_tracker.connectivity,
        'Path Efficiency': evaluation_tracker.path_efficiency,
        'Population Coverage': evaluation_tracker.population,
        'Budget Used (%)': [b * 100 for b in evaluation_tracker.budget_used]
    }
    
    # Create subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3*len(metrics)), sharex=True)
    episodes = np.arange(1, len(evaluation_tracker.rewards) + 1)
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(episodes, values, marker='o', linestyle='-', markersize=6)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.legend()
    
    # Set common x label
    axes[-1].set_xlabel("Evaluation Episode")
    
    # Add title
    title = "Evaluation Metrics"
    if city_name:
        title += f" - {city_name}"
    fig.suptitle(title)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, "evaluation_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation metrics plot saved to: {save_path}")

def main():
    """Main function to demonstrate the bike path RL model with reward plotting."""
    import argparse
    import time
    import multiprocessing

    max_cpu_count = multiprocessing.cpu_count()
    default_n_envs = max(1, max_cpu_count-1)
    default_n_steps = 2048
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate a RL model for bike path planning')
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument('--city', type=str, 
                        help='City name to analyze')
    location_group.add_argument('--bbox', type=float, nargs=4, metavar=('NORTH', 'SOUTH', 'EAST', 'WEST'),
                        help='Bounding box coordinates (north, south, east, west)')
    parser.add_argument('--budget', type=float, default=100000, 
                        help='Budget for bike path construction (default: 10000)')
    parser.add_argument('--timesteps', type=int, default=10240, 
                        help='Training timesteps (default: 10240)')
    parser.add_argument('--eval_episodes', type=int, default=5, 
                        help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--skip_training', action='store_true', 
                        help='Skip training and load existing model')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to existing model (for --skip_training)')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Training device (default: cpu)')
    parser.add_argument('--n_envs', type=int, default=default_n_envs, 
                        help='Number of environments (default: max-1 cpu count))')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable reward plotting')
    
    args = parser.parse_args()

    batch_size = default_n_steps * args.n_envs
    num_updates = math.ceil(args.timesteps / batch_size)
    adjusted_timesteps = num_updates * batch_size
    
    start_time = time.time()

    print("\n" + "="*70)
    print(f"BIKE PATH REINFORCEMENT LEARNING MODEL")
    print("="*70)
    print(f"City: {args.city}")
    print(f"Budget: {args.budget}")
    print(f"Bbox: {args.bbox}")
    print(f"Requested timesteps: {args.timesteps}")
    print(f"Adjusted timesteps: {adjusted_timesteps}")
    print(f"Using {args.n_envs} environments (of {max_cpu_count} available cores))")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print("="*70 + "\n")
    
    # Setup run directory for organizing outputs
    city_name = args.city if args.city else f"BBox_{args.bbox}" if args.bbox else "Unknown"
    setup_run_directory(city_name)
    
    if args.skip_training and args.model_path:
        from stable_baselines3 import PPO
        print(f"Loading existing model from {args.model_path}...")
        model = PPO.load(args.model_path)
    else:
        model = train_rl_model(
            city_name=args.city, 
            bbox=args.bbox, 
            budget=args.budget, 
            total_timesteps=adjusted_timesteps, 
            device=args.device, 
            n_envs=args.n_envs,
            plot_rewards=not args.no_plots
        )
        
        # Save the final model to the run directory
        if RUN_FIGURES_DIR:
            model_path = os.path.join(RUN_FIGURES_DIR, "final_model.zip")
            model.save(model_path)
            print(f"Final model saved to {model_path}")
    
    best_paths, best_state = evaluate_and_visualize(
        model, 
        city_name=args.city, 
        bbox=args.bbox, 
        budget=args.budget,
        num_evaluations=args.eval_episodes
    )
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save a summary file with run information
    if RUN_FIGURES_DIR:
        summary_path = os.path.join(RUN_FIGURES_DIR, "run_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"BIKE PATH REINFORCEMENT LEARNING RUN SUMMARY\n")
            f.write(f"=======================================\n")
            f.write(f"Run ID: {RUN_ID}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"City: {args.city}\n")
            f.write(f"Bbox: {args.bbox}\n")
            f.write(f"Budget: {args.budget}\n")
            f.write(f"Timesteps: {adjusted_timesteps}\n")
            f.write(f"Environments: {args.n_envs}\n")
            f.write(f"Evaluation episodes: {args.eval_episodes}\n")
            f.write(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
            f.write(f"=======================================\n")
        print(f"Run summary saved to {summary_path}")
    
    return best_paths, best_state

if __name__ == "__main__":
    main()