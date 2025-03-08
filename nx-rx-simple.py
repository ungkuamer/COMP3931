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
import itertools
from datetime import datetime, timedelta
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback


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
    
    def __init__(self, city_name, budget=1000, edge_cost_factor=10):
        super(BikePathEnv, self).__init__()
        
        print(f"Initializing environment for {city_name}")
        
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
        """Identify roads that don't have bike infrastructure with improved validation."""
        candidates = []
        #print("Finding candidate edges...")
        
        # Get existing bike edges
        bike_edges = set((u, v) for u, v, _ in self.original_graph.edges(keys=True))
        validation_errors = 0
        
        # Iterate through walkable edges
        for u, v, data in self.walk_graph.edges(data=True):
            try:
                if (u, v) not in bike_edges and (v, u) not in bike_edges:
                    # Validate road type
                    road_type = data.get('highway', '')
                    if isinstance(road_type, list):
                        road_type = road_type[0] if road_type else ''
                    
                    suitable_types = {'residential', 'tertiary', 'secondary', 'primary', 
                                    'unclassified', 'living_street'}
                    
                    if road_type in suitable_types:
                        # Validate node coordinates
                        u_coords = (self.walk_graph.nodes[u].get('y'), self.walk_graph.nodes[u].get('x'))
                        v_coords = (self.walk_graph.nodes[v].get('y'), self.walk_graph.nodes[v].get('x'))
                        
                        if all(u_coords) and all(v_coords):
                            # Calculate or validate length
                            if 'length' not in data:
                                data['length'] = ox.distance.great_circle(u_coords, v_coords)
                            
                            # Additional validation
                            if data['length'] > 0 and data['length'] < 500:  # Max 500m segments
                                candidates.append((u, v, data))
                            
                        else:
                            validation_errors += 1
                            if validation_errors < 10:  # Limit error messages
                                print(f"Warning: Missing coordinates for edge ({u}, {v})")
            except Exception as e:
                validation_errors += 1
                if validation_errors < 10:
                    print(f"Error processing edge ({u}, {v}): {e}")
        
        #print(f"Found {len(candidates)} valid candidate edges")
        if validation_errors > 0:
            print(f"Encountered {validation_errors} validation errors during candidate edge search")
        
        return candidates
    
    def _calculate_connectivity(self):
        """Calculate connectivity metrics using rustworkx's built-in parallel functionality."""
        try:
            # Verify graph has nodes and edges
            if len(self.graph.nodes()) == 0 or len(self.graph.edges()) == 0:
                print("Warning: Graph has no nodes or edges")
                return 0.0, 0.0
                
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
            
            # Calculate average clustering coefficient using rustworkx's built-in function
            # rx.transitivity calculates the global clustering coefficient
            clustering = rx.transitivity(rx_subgraph)
            if clustering is None:
                clustering = 0.0
                
            #print(f"Successfully calculated clustering: {clustering:.4f}")
            
            # Calculate average path length using rustworkx's built-in function
            # Use a sampling approach for large graphs
            if rx_subgraph.num_nodes() > 100:
                sampled_nodes = random.sample(range(rx_subgraph.num_nodes()), 100)
                node_pairs = list(itertools.combinations(sampled_nodes, 2))
            else:
                node_pairs = list(itertools.combinations(range(rx_subgraph.num_nodes()), 2))
            
            # Define weight function for rustworkx
            def weight_fn(edge_data):
                return edge_data.get('length', 1.0)
            
            # Use rustworkx's all-pairs shortest path function
            path_lengths = []
            
            # Calculate path lengths for each pair in the sample
            for u, v in node_pairs:
                try:
                    # Use dijkstra_shortest_path_lengths which returns a dictionary
                    path_lengths_dict = rx.dijkstra_shortest_path_lengths(rx_subgraph, u, weight_fn, goal=v)
                    if v in path_lengths_dict and path_lengths_dict[v] > 0:
                        path_lengths.append(path_lengths_dict[v])
                except (rx.NoPathFound, Exception) as e:
                    print(f"Error calculating path length between {u} and {v}: {e}")
                    exit()
            
            # Calculate metrics
            if path_lengths:
                avg_path_length = np.mean(path_lengths)
                #print(f"Successfully calculated {len(path_lengths)} path lengths")
                #print(f"Average path length: {avg_path_length:.1f}m")
            else:
                print("Warning: No valid path lengths calculated")
                avg_path_length = float('inf')
            
            # Normalize metrics
            normalized_clustering = min(max(clustering, 0.0), 1.0)
            if avg_path_length == float('inf'):
                normalized_path = 0.0
            else:
                normalized_path = 1.0 / (1.0 + avg_path_length/1000)
            
            #print(f"Final metrics - clustering: {normalized_clustering:.4f}, path_efficiency: {normalized_path:.4f}")
            
            return normalized_clustering, normalized_path
            
        except Exception as e:
            print(f"Connectivity calculation failed: {e}")
            return 0.0, 0.0
    

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
            
            # Validate metrics
            state = np.array([
                max(0.0, min(1.0, connectivity)),
                max(0.0, min(1.0, path_efficiency)),
                max(0.0, min(1.0, pop_served)),
                max(0.0, min(1.0, norm_budget))
            ], dtype=np.float32)
            
            '''
            # Log state if it changed significantly
            if hasattr(self, 'state'):
                change = np.abs(state - self.state).max()
                if change > 0.1:  # Log significant changes
                    print(f"Significant state change detected (delta={change:.4f}):")
                    print(f"Old state: {[f'{x:.4f}' for x in self.state]}")
                    print(f"New state: {[f'{x:.4f}' for x in state]}")
            '''

            return state
            
        except Exception as e:
            print(f"Error calculating state: {e}")
            return np.zeros(4, dtype=np.float32)
    
    def step(self, action):
        """Take an action with improved error handling and state updates."""
        self.episode_steps += 1
        self.total_steps += 1
        
        # Validate action
        if action >= len(self.candidate_edges):
            #print(f"Warning: Invalid action {action} (max allowed: {len(self.candidate_edges)-1})")
            return self.state, -10, True, False, {}
        
        # Get selected edge
        u, v, data = self.candidate_edges[action]
        
        # Validate edge cost
        cost = data.get('length', 0) * self.edge_cost_factor
        if cost <= 0:
            #print(f"Warning: Invalid edge cost {cost} for edge ({u}, {v})")
            return self.state, -5, False, False, {}
        
        # Check budget
        if cost > self.budget:
            #print(f"Not enough budget for edge ({u}, {v}). Cost: {cost:.1f}, Budget: {self.budget:.1f}")
            self.candidate_edges.pop(action)
            
            # Check if we're done
            min_edge_cost = min([d['length'] * self.edge_cost_factor for _, _, d in self.candidate_edges]) if self.candidate_edges else float('inf')
            done = (self.budget < min_edge_cost) or len(self.candidate_edges) == 0
            
            return self.state, -1, done, False, {}
        
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
                
            self.added_paths.append((u, v, edge_data['length']))
        
        # Calculate new state and reward
        old_state = self.state
        self.state = self._get_state()
        
        # Calculate reward with more weight on connectivity improvements
        state_improvement = [
            (new - old) * weight for new, old, weight in 
            zip(self.state[:3], old_state[:3], [0.4, 0.3, 0.3])  # Weights for each metric
        ]
        reward = sum(state_improvement) * 100
        
        # Add reward for efficient budget use
        budget_efficiency = cost / (data['length'] + 1)  # Lower cost per meter is better
        reward += max(0, (1 - budget_efficiency/10) * 10)  # Scale and cap budget efficiency reward
        
        self.episode_reward += reward
        
        # Remove used edge and check completion
        self.candidate_edges.pop(action)
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
            "action_index": action,
            "edge_cost": cost,
            "edge_length": data['length'],
            "actions_remaining": len(self.candidate_edges)
        }
        
        # Log progress
        '''
        if self.episode_steps % 10 == 0 or done:
            print(f"\nStep {self.episode_steps}:")
            print(f"Added path: ({u},{v}), length={data['length']:.1f}m, cost={cost:.1f}")
            print(f"State: {[f'{x:.4f}' for x in self.state]}")
            print(f"Reward: {reward:.2f}")
            print(f"Budget: {self.budget:.1f}/{self.initial_budget:.1f}")
            print(f"Remaining edges: {len(self.candidate_edges)}")
        '''

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
            #print(f"\nReset for episode {self.episode_count}")
            #print(f"Previous episode: reward={self.episode_rewards[-1]:.2f}, steps={self.episode_lengths[-1]}")
            #print(f"Running average reward: {avg_reward:.2f}")
        
        # Reset graph and validate
        self.graph = self.original_graph.copy()
        self._validate_graph_attributes()
        
        # Reset rustworkx graph
        self.rx_graph, self.rx_node_map = nx_to_rx(self.graph)
        
        # Reset other variables
        self.budget = self.initial_budget
        self.candidate_edges = self._get_candidate_edges()
        self.added_paths = []
        self.episode_steps = 0
        self.episode_reward = 0
        
        # Get initial state
        self.state = self._get_state()
        #print(f"Initial state: {[f'{x:.4f}' for x in self.state]}")
        
        return self.state, {}
    
    def render(self, mode='human'):
        """Render with improved visualization and error handling."""
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
            
            # Add metrics to title
            conn, path_eff, pop, budget_remain = self.state
            plt.title(
                f"Bike Network Plan - {self.city_name}\n"
                f"Connectivity: {conn:.3f} | Path Efficiency: {path_eff:.3f} | "
                f"Population Served: {pop:.3f}\n"
                f"Budget: {self.budget:.0f}/{self.initial_budget:.0f} "
                f"({(1-budget_remain)*100:.1f}% used)"
            )
            
            # Save figure
            os.makedirs("./bike_path_figures", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = f"./bike_path_figures/{self.city_name.replace(', ', '_')}_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_path}")
            
            if mode == 'human':
                plt.show()
            else:
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
    Minimal progress callback showing only a progress bar during training.
    """
    def __init__(self, total_timesteps):
        super(TrainingProgressCallback, self).__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.training_start = None
        self.episode_rewards = []
        
    def _on_training_start(self):
        self.training_start = time.time()
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")
        
    def _on_step(self):
        if self.pbar:
            steps_done = self.num_timesteps - self.pbar.n
            self.pbar.update(steps_done)
            
            if "episode" in self.locals["infos"][0]:
                reward = self.locals["infos"][0]["episode"]["r"]
                self.episode_rewards.append(reward)
                avg_reward = np.mean(self.episode_rewards[-100:])
                self.pbar.set_description(f"Training (reward: {avg_reward:.2f})")
        
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

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

def train_rl_model(city_name='Amsterdam, Netherlands', budget=10000, total_timesteps=50000):
    """Train RL model with minimal output."""
    env = BikePathEnv(city_name=city_name, budget=budget)
    env = DummyVecEnv([lambda: env])
    
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
        device='auto'
    )
    
    # Setup callbacks
    progress_callback = TrainingProgressCallback(total_timesteps)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1000),
        save_path="./checkpoints/",
        name_prefix="bike_model",
        verbose=0
    )
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, checkpoint_callback]
    )
    
    return model


def evaluate_and_visualize(model, city_name=None, budget=10000, num_evaluations=5):
    """Evaluate model with improved progress tracking."""
    # Initialize evaluation tracker
    tracker = EvaluationTracker(num_evaluations)
    tracker.start_evaluation()
    
    try:
        for _ in range(num_evaluations):
            # Create fresh environment
            env = BikePathEnv(city_name=city_name, budget=budget)
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
                'env': env
            }
            
            tracker.update_episode(metrics)
    
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise
    finally:
        tracker.close()
    
    # Print evaluation summary
    tracker.print_summary()
    
    # Visualize best solution
    if tracker.best_solution:
        best_env = tracker.best_solution['env']
        best_env.render(mode='human')
        
        # Export results if available
        try:
            gdf = best_env.get_edge_gdf()
            if gdf is not None:
                safe_name = city_name.replace(', ', '_').replace(' ', '_')
                gdf.to_file(f"suggested_bike_paths_{safe_name}.geojson", driver='GeoJSON')
        except Exception as e:
            print(f"Error exporting results: {e}")
    
    return tracker.best_solution['paths'], tracker.best_solution['env'].state if tracker.best_solution else (None, None)

def main():
    """Main function to demonstrate the bike path RL model."""
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate a RL model for bike path planning')
    parser.add_argument('--city', type=str, default='Colmar, France', 
                        help='City name to analyze (default: Amsterdam, Netherlands)')
    parser.add_argument('--budget', type=float, default=10000, 
                        help='Budget for bike path construction (default: 10000)')
    parser.add_argument('--timesteps', type=int, default=10000, 
                        help='Training timesteps (default: 10000)')
    parser.add_argument('--eval_episodes', type=int, default=3, 
                        help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--skip_training', action='store_true', 
                        help='Skip training and load existing model')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to existing model (for --skip_training)')
    
    args = parser.parse_args()
    
    start_time = time.time()

    print("\n" + "="*70)
    print(f"BIKE PATH REINFORCEMENT LEARNING MODEL")
    print("="*70)
    print(f"City: {args.city}")
    print(f"Budget: {args.budget}")
    print(f"Training timesteps: {args.timesteps}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print("="*70 + "\n")
    
    if args.skip_training and args.model_path:
        from stable_baselines3 import PPO
        print(f"Loading existing model from {args.model_path}...")
        model = PPO.load(args.model_path)
    else:
        model = train_rl_model(city_name=args.city, budget=args.budget, total_timesteps=args.timesteps)
    
    best_paths, best_state = evaluate_and_visualize(
        model, 
        city_name=args.city, 
        budget=args.budget,
        num_evaluations=args.eval_episodes
    )
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return best_paths, best_state

if __name__ == "__main__":
    main()