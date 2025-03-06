import osmnx as ox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import gymnasium as gym
from gymnasium import spaces
import random
import time
import sys
import os
import igraph as ig
from multiprocessing import Pool, cpu_count
import itertools
from datetime import datetime
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

class BikePathEnv(gym.Env):
    """
    Custom Environment for optimizing bike path additions to a city network.
    
    This environment uses real city street networks from OSMnx and allows an RL agent
    to suggest optimal bike path additions based on metrics like connectivity, safety,
    and route directness. Uses igraph for enhanced performance.
    """
    
    def __init__(self, city_name, budget=1000, edge_cost_factor=10):
        super(BikePathEnv, self).__init__()
        
        print(f"Initializing environment for {city_name}")
        
        # Load the city graph using OSMnx but convert to igraph
        self.nx_original_graph = ox.graph_from_place(city_name, network_type='bike')
        self.city_name = city_name
        
        # Convert NetworkX graph to igraph
        self.original_graph = self._convert_nx_to_igraph(self.nx_original_graph)
        
        # Create a working copy of the graph
        self.graph = self.original_graph.copy()
        
        # Get all roads that don't already have bike infrastructure
        self.nx_walk_graph = ox.graph_from_place(city_name, network_type='walk')
        self.walk_graph = self._convert_nx_to_igraph(self.nx_walk_graph)
        
        # Add better validation of graphs
        print(f"Original bike graph: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
        print(f"Walk graph: {self.walk_graph.vcount()} nodes, {self.walk_graph.ecount()} edges")
        
        # Verify graph attributes
        self._validate_graph_attributes()
        
        # Identify candidate edges for bike path addition
        self.candidate_edges = self._get_candidate_edges()
        print(f"Found {len(self.candidate_edges)} candidate edges for bike paths")
        
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
        
        # Store node mapping for later reference
        self.node_map = {}  # Maps original OSM IDs to igraph vertex indices
        
        # Print initial state
        print("Initial state:", self.state)

    def _convert_nx_to_igraph(self, nx_graph):
        """Convert NetworkX graph to igraph."""
        # Create a new igraph Graph
        g = ig.Graph(directed=nx_graph.is_directed())
        
        # Reset the node map
        self.node_map = {}
        
        # Add vertices with attributes
        for i, (node, attrs) in enumerate(nx_graph.nodes(data=True)):
            # Store the mapping from original ID to vertex index
            self.node_map[node] = i
            
            # Add vertex with attributes
            g.add_vertex(name=str(node))
            
            # Add node attributes
            for k, v in attrs.items():
                g.vs[i][k] = v
        
        # Add edges with attributes
        for u, v, key, attrs in nx_graph.edges(data=True, keys=True):
            # Get corresponding vertex indices
            u_idx = self.node_map[u]
            v_idx = self.node_map[v]
            
            # Add edge
            edge_id = g.add_edge(u_idx, v_idx)
            
            # Add edge attributes
            for k, v in attrs.items():
                g.es[edge_id][k] = v
            
            # Add the key attribute from MultiGraph
            g.es[edge_id]['key'] = key
        
        return g

    def _validate_graph_attributes(self):
        """Validate and fix graph attributes."""
        print("Validating graph attributes...")
        
        # Check and fix node attributes
        for v in self.graph.vs:
            if 'x' not in v.attributes() or 'y' not in v.attributes():
                print(f"Warning: Node {v['name']} missing coordinates")
                # Try to find in walk graph
                try:
                    walk_node_idx = self.node_map.get(v['name'])
                    if walk_node_idx is not None:
                        walk_v = self.walk_graph.vs[walk_node_idx]
                        if 'x' in walk_v.attributes() and 'y' in walk_v.attributes():
                            v['x'] = walk_v['x']
                            v['y'] = walk_v['y']
                except:
                    pass
        
        # Check and fix edge attributes
        for e in self.graph.es:
            if 'length' not in e.attributes() or e['length'] <= 0:
                print(f"Warning: Edge ({e.source}, {e.target}) missing length attribute")
                # Calculate length using node coordinates
                try:
                    source_v = self.graph.vs[e.source]
                    target_v = self.graph.vs[e.target]
                    u_coords = (source_v['y'], source_v['x'])
                    v_coords = (target_v['y'], target_v['x'])
                    e['length'] = ox.distance.great_circle(u_coords, v_coords)
                except Exception as err:
                    print(f"Error calculating length for edge ({e.source}, {e.target}): {err}")
        
    def _calculate_node_clustering(self, args):
        """Calculate clustering coefficient for a single node."""
        G, node_idx = args
        try:
            # This is now using igraph's built-in transitivity (clustering) function
            return G.transitivity_local_undirected([node_idx])[0]
        except Exception as e:
            print(f"Error calculating clustering for node {node_idx}: {e}")
            return 0.0
        
    def _calculate_path_lengths_chunk(self, args):
        """Calculate path lengths for a chunk of node pairs."""
        G, node_pairs = args
        lengths = []
        for u_idx, v_idx in node_pairs:
            try:
                # igraph's shortest_paths returns a matrix, we need [0][0] for first pair
                path_length = G.shortest_paths(source=u_idx, target=v_idx, weights='length')[0][0]
                if path_length > 0:
                    lengths.append(path_length)
            except Exception as e:
                continue
        return lengths
               
    def _get_candidate_edges(self):
        """Identify roads that don't have bike infrastructure with improved validation."""
        candidates = []
        print("Finding candidate edges...")
        
        # Prepare a set of existing bike edges for quick lookup
        # We need to convert from igraph edge indices to (source, target) tuples
        bike_edges = set()
        for e in self.original_graph.es:
            source_name = self.original_graph.vs[e.source]['name']
            target_name = self.original_graph.vs[e.target]['name']
            bike_edges.add((source_name, target_name))
            bike_edges.add((target_name, source_name))  # Add reverse direction
            
        validation_errors = 0
        
        # Iterate through walkable edges
        for e in self.walk_graph.es:
            try:
                source_name = self.walk_graph.vs[e.source]['name']
                target_name = self.walk_graph.vs[e.target]['name']
                
                if (source_name, target_name) not in bike_edges and (target_name, source_name) not in bike_edges:
                    # Get edge attributes
                    data = e.attributes()
                    
                    # Validate road type
                    road_type = data.get('highway', '')
                    if isinstance(road_type, list):
                        road_type = road_type[0] if road_type else ''
                    
                    suitable_types = {'residential', 'tertiary', 'secondary', 'primary', 
                                    'unclassified', 'living_street'}
                    
                    if road_type in suitable_types:
                        # Validate node coordinates
                        source_v = self.walk_graph.vs[e.source]
                        target_v = self.walk_graph.vs[e.target]
                        
                        u_coords = (source_v.get('y'), source_v.get('x'))
                        v_coords = (target_v.get('y'), target_v.get('x'))
                        
                        if all(u_coords) and all(v_coords):
                            # Calculate or validate length
                            if 'length' not in data:
                                data['length'] = ox.distance.great_circle(u_coords, v_coords)
                            
                            # Additional validation
                            if data['length'] > 0 and data['length'] < 500:  # Max 500m segments
                                candidates.append((e.source, e.target, data))
                        else:
                            validation_errors += 1
                            if validation_errors < 10:  # Limit error messages
                                print(f"Warning: Missing coordinates for edge ({e.source}, {e.target})")
            except Exception as e:
                validation_errors += 1
                if validation_errors < 10:
                    print(f"Error processing edge: {e}")
        
        print(f"Found {len(candidates)} valid candidate edges")
        if validation_errors > 0:
            print(f"Encountered {validation_errors} validation errors during candidate edge search")
        
        return candidates
    
    def _calculate_connectivity(self):
        """Calculate connectivity metrics using parallel processing with igraph."""
        try:
            # Verify graph has nodes and edges
            if self.graph.vcount() == 0 or self.graph.ecount() == 0:
                print("Warning: Graph has no nodes or edges")
                return 0.0, 0.0
            
            # Get largest connected component using igraph
            components = self.graph.clusters()
            largest_cc = components.giant()
            
            if largest_cc.vcount() == 0:
                print("Warning: Largest component has no nodes")
                return 0.0, 0.0
            
            # Number of processes to use (leave one core free)
            n_processes = max(1, cpu_count() - 1)
            print(f"Using {n_processes} processes for parallel computation")
            
            # Calculate average clustering using igraph's built-in function
            clustering = largest_cc.transitivity_avglocal_undirected(mode="zero")
            print(f"Successfully calculated clustering: {clustering:.4f}")
            
            # Parallel path length calculation
            if largest_cc.vcount() > 100:
                sampled_nodes = random.sample(range(largest_cc.vcount()), 100)
                node_pairs = list(itertools.combinations(sampled_nodes, 2))
            else:
                node_pairs = list(itertools.combinations(range(largest_cc.vcount()), 2))
            
            # Split node pairs into chunks for parallel processing
            chunk_size = max(1, len(node_pairs) // (n_processes * 4))
            chunks = [node_pairs[i:i + chunk_size] for i in range(0, len(node_pairs), chunk_size)]
            
            with Pool(n_processes) as pool:
                # Prepare arguments for each chunk
                chunk_args = [(largest_cc, chunk) for chunk in chunks]
                
                # Calculate path lengths in parallel
                path_lengths_chunks = pool.map(self._calculate_path_lengths_chunk, chunk_args)
                
                # Combine results
                path_lengths = list(itertools.chain.from_iterable(path_lengths_chunks))
            
            # Calculate metrics
            if path_lengths:
                avg_path_length = np.mean(path_lengths)
                print(f"Successfully calculated {len(path_lengths)} path lengths")
                print(f"Average path length: {avg_path_length:.1f}m")
            else:
                print("Warning: No valid path lengths calculated")
                avg_path_length = float('inf')
            
            # Normalize metrics
            normalized_clustering = min(max(clustering, 0.0), 1.0)
            if avg_path_length == float('inf'):
                normalized_path = 0.0
            else:
                normalized_path = 1.0 / (1.0 + avg_path_length/1000)
            
            print(f"Final metrics - clustering: {normalized_clustering:.4f}, path_efficiency: {normalized_path:.4f}")
            
            return normalized_clustering, normalized_path
            
        except Exception as e:
            print(f"Connectivity calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_population_served(self):
        """Calculate population served with improved metrics."""
        try:
            if self.graph.vcount() == 0:
                return 0.0
            
            # Calculate based on both network coverage and connectivity
            bike_nodes = self.graph.vcount()
            all_nodes = self.walk_graph.vcount()
            
            # Basic coverage ratio
            coverage_ratio = bike_nodes / max(1, all_nodes)
            
            # Get connected components info
            components = self.graph.clusters()
            
            if len(components) == 0:
                return 0.0
                
            # Calculate component size ratio - use giant component
            largest_component_size = max(components.sizes())
            component_ratio = largest_component_size / max(1, bike_nodes)
            
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
            
            # Log state if it changed significantly
            if hasattr(self, 'state'):
                change = np.abs(state - self.state).max()
                if change > 0.1:  # Log significant changes
                    print(f"Significant state change detected (delta={change:.4f}):")
                    print(f"Old state: {[f'{x:.4f}' for x in self.state]}")
                    print(f"New state: {[f'{x:.4f}' for x in state]}")
            
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
            print(f"Warning: Invalid action {action} (max allowed: {len(self.candidate_edges)-1})")
            return self.state, -10, True, False, {}
        
        # Get selected edge
        u, v, data = self.candidate_edges[action]
        
        # Validate edge cost
        cost = data.get('length', 0) * self.edge_cost_factor
        if cost <= 0:
            print(f"Warning: Invalid edge cost {cost} for edge ({u}, {v})")
            return self.state, -5, False, False, {}
        
        # Check budget
        if cost > self.budget:
            print(f"Not enough budget for edge ({u}, {v}). Cost: {cost:.1f}, Budget: {self.budget:.1f}")
            self.candidate_edges.pop(action)
            
            # Check if we're done
            min_edge_cost = min([d['length'] * self.edge_cost_factor for _, _, d in self.candidate_edges]) if self.candidate_edges else float('inf')
            done = (self.budget < min_edge_cost) or len(self.candidate_edges) == 0
            
            return self.state, -1, done, False, {}
        
        # Update budget
        self.budget -= cost
        
        # Add edge to graph
        # Check if edge already exists
        edge_id = self.graph.get_eid(u, v, directed=False, error=False)
        if edge_id == -1:  # Edge doesn't exist
            # Ensure nodes have proper attributes from walk graph
            for node_idx, attr_src in [(u, self.walk_graph.vs[u]), (v, self.walk_graph.vs[v])]:
                for attr_name in ['x', 'y']:
                    if attr_name not in self.graph.vs[node_idx].attributes() or not self.graph.vs[node_idx][attr_name]:
                        if attr_name in attr_src.attributes() and attr_src[attr_name]:
                            self.graph.vs[node_idx][attr_name] = attr_src[attr_name]
            
            # Add the edge with attributes
            edge_data = data.copy()
            edge_data['bike_lane'] = 'yes'  # Mark as bike path
            
            # Add the edge
            edge_id = self.graph.add_edge(u, v)
            
            # Add attributes to the edge
            for k, v in edge_data.items():
                self.graph.es[edge_id][k] = v
                
            # Ensure length attribute exists
            if 'length' not in edge_data or edge_data['length'] <= 0:
                source_v = self.graph.vs[u]
                target_v = self.graph.vs[v]
                u_coords = (source_v['y'], source_v['x'])
                v_coords = (target_v['y'], target_v['x'])
                self.graph.es[edge_id]['length'] = ox.distance.great_circle(u_coords, v_coords)
            
            self.added_paths.append((u, v, self.graph.es[edge_id]['length']))
        
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
        if self.episode_steps % 10 == 0 or done:
            print(f"\nStep {self.episode_steps}:")
            print(f"Added path: ({u},{v}), length={data['length']:.1f}m, cost={cost:.1f}")
            print(f"State: {[f'{x:.4f}' for x in self.state]}")
            print(f"Reward: {reward:.2f}")
            print(f"Budget: {self.budget:.1f}/{self.initial_budget:.1f}")
            print(f"Remaining edges: {len(self.candidate_edges)}")
        
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
            print(f"\nReset for episode {self.episode_count}")
            print(f"Previous episode: reward={self.episode_rewards[-1]:.2f}, steps={self.episode_lengths[-1]}")
            print(f"Running average reward: {avg_reward:.2f}")
        
        # Reset graph and validate
        self.graph = self.original_graph.copy()
        self._validate_graph_attributes()
        
        # Reset other variables
        self.budget = self.initial_budget
        self.candidate_edges = self._get_candidate_edges()
        self.added_paths = []
        self.episode_steps = 0
        self.episode_reward = 0
        
        # Get initial state
        self.state = self._get_state()
        print(f"Initial state: {[f'{x:.4f}' for x in self.state]}")
        
        return self.state, {}
    
    def render(self, mode='human'):
        """Render with improved visualization and error handling."""
        try:
            fig, ax = plt.subplots(figsize=(15, 12))
            
            # Convert igraph back to NetworkX for plotting with OSMnx
            G_base = self._convert_igraph_to_networkx(self.original_graph)
            
            # Plot base network
            if len(G_base.edges()) > 0:
                ox.plot_graph(G_base, ax=ax, node_size=0, 
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
    
    def _convert_igraph_to_networkx(self, ig_graph):
        """Convert igraph back to NetworkX for visualization."""
        import networkx as nx
        
        # Create empty networkx graph
        if ig_graph.is_directed():
            nx_graph = nx.MultiDiGraph()
        else:
            nx_graph = nx.MultiGraph()
        
        # Add nodes with attributes
        for v in ig_graph.vs:
            node_attrs = {k: v[k] for k in v.attributes() if k != 'name'}
            nx_graph.add_node(v['name'], **node_attrs)
        
        # Add edges with attributes
        for e in ig_graph.es:
            source_name = ig_graph.vs[e.source]['name']
            target_name = ig_graph.vs[e.target]['name']
            edge_attrs = {k: e[k] for k in e.attributes() if k != 'key'}
            key = e.get('key', 0)
            nx_graph.add_edge(source_name, target_name, key=key, **edge_attrs)
        
        return nx_graph
        
    def get_edge_gdf(self):
        """Convert added paths to GeoDataFrame with improved error handling."""
        try:
            edges = []
            for u, v, length in self.added_paths:
                try:
                    u_data = self.graph.vs[u].attributes()
                    v_data = self.graph.vs[v].attributes()
                    
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


def train_rl_model(city_name='Amsterdam, Netherlands', budget=10000, total_timesteps=50000):
    """Train RL model with improved monitoring and error handling."""
    print(f"\n{'='*50}")
    print(f"Starting training for {city_name}")
    print(f"Budget: {budget}, Timesteps: {total_timesteps}")
    print(f"{'='*50}\n")
    
    try:
        # Create and validate environment
        env = BikePathEnv(city_name=city_name, budget=budget)
        
        # Setup logging
        log_dir = f"./bike_path_logs/{city_name.replace(', ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create vectorized environment
        env = DummyVecEnv([lambda: env])
        
        # Initialize model with improved parameters
        model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=0.0003,  # Slightly lower learning rate
            n_steps=2048,  # Increased batch size
            batch_size=64,
            n_epochs=10,  # More training epochs
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            device='cpu'
        )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 10, 1000),
            save_path=f"{log_dir}/checkpoints/",
            name_prefix="bike_model",
            verbose=1
        )
        
        progress_callback = TrainingProgressCallback(verbose=1, log_freq=1000)
        
        # Train model with progress tracking
        print("\nStarting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback]
        )
        
        # Save final model
        final_model_path = f"{log_dir}/final_model"
        model.save(final_model_path)
        print(f"\nTraining complete! Model saved to {final_model_path}")
        
        return model
        
    except Exception as e:
        print(f"Error in training: {e}")
        raise


def evaluate_and_visualize(model, city_name=None, north=None, south=None, east=None, west=None, 
                          budget=10000, num_evaluations=5, export_shapefile=True):
    """Evaluate the trained model and visualize results."""
    print(f"\n{'='*50}")
    if city_name:
        print(f"Evaluating model for {city_name}")
    else:
        print(f"Evaluating model for bounded area ({north},{west} to {south},{east})")
    print(f"{'='*50}")
    
    best_reward = -float('inf')
    best_paths = None
    best_state = None
    best_env = None
    
    # Tracking metrics across all evaluations
    all_rewards = []
    all_num_paths = []
    all_connectivity = []
    all_path_efficiency = []
    all_population = []
    
    print(f"\nRunning {num_evaluations} evaluation episodes...\n")
    
    for i in tqdm(range(num_evaluations), desc="Evaluation Progress"):
        # Create a fresh environment for each evaluation
        env = BikePathEnv(city_name=city_name, budget=budget)
        
        # Reset environment
        obs, _ = env.reset(seed=i)  # Use episode index as seed for reproducibility
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        # Track episode start time
        start_time = time.time()
        
        # Track selected actions for debugging
        action_history = []
        
        # Add a safety counter to prevent infinite loops
        safety_counter = 0
        max_safety_count = 1000  # Maximum number of iterations to prevent infinite loops
        
        # Run episode
        while not done and not truncated and safety_counter < max_safety_count:
            safety_counter += 1
            
            # Get action from model (reshape for vectorized environment)
            action, _ = model.predict(np.array(obs).reshape(1, -1), deterministic=True)
            action = int(action[0])  # Get scalar action from vectorized output
            
            # Debug output for every action
            action_history.append(action)
            
            # Check if action is valid (within action space)
            if action >= env.action_space.n:
                print(f"Warning: Model predicted invalid action {action}, max allowed is {env.action_space.n-1}")
                # Choose a random valid action instead
                action = np.random.randint(0, env.action_space.n)
                print(f"Choosing random action {action} instead")
            
            # Debug info every few steps
            if steps % 5 == 0:
                print(f"Step {steps}: Selecting action {action} out of {env.action_space.n} possible actions")
                if len(action_history) > 5:
                    print(f"Recent action history: {action_history[-5:]}")
            
            # Take step in environment
            old_obs = obs.copy()
            obs, reward, done, truncated, info = env.step(action)
            
            # Check if state changed
            if np.array_equal(old_obs, obs) and not done:
                print(f"Warning: State did not change after action {action}. This might indicate a problem.")
                
                # If we're stuck in a loop where state doesn't change, try a random action
                if safety_counter > 20 and len(set(action_history[-10:])) <= 2:
                    print("Detected potential action loop. Trying random action.")
                    rand_action = np.random.randint(0, env.action_space.n)
                    if rand_action != action:
                        action = rand_action
                        print(f"Selected random action {action}")
                        obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # If safety counter is getting high, print debug info
            if safety_counter > 100 and safety_counter % 50 == 0:
                print(f"Safety counter: {safety_counter}. Still running. Actions remaining: {env.action_space.n}")
                print(f"Budget remaining: {env.budget:.1f}")
        
        # Check if we hit safety limit
        if safety_counter >= max_safety_count:
            print(f"Warning: Episode terminated due to safety counter limit ({max_safety_count}). This indicates a potential issue.")
        
        # Calculate episode duration
        duration = time.time() - start_time
        
        # Get state variables
        conn, path_eff, pop, remaining_budget = env.state
        
        # Store metrics
        all_rewards.append(total_reward)
        all_num_paths.append(len(env.added_paths))
        all_connectivity.append(conn)
        all_path_efficiency.append(path_eff)
        all_population.append(pop)
        
        # Count unique actions taken
        unique_actions = len(set(action_history))
        
        # Display progress with detailed metrics
        print(f"Episode {i+1}: Reward={total_reward:.2f}, Paths={len(env.added_paths)}, " +
              f"Time={duration:.2f}s, Steps={steps}")
        print(f"  - Actions taken: {steps}, Unique actions: {unique_actions} (out of {len(action_history)} total)")
        print(f"  - Connectivity: {conn:.4f}, Path Efficiency: {path_eff:.4f}, Population: {pop:.4f}")
        print(f"  - Budget Used: {(1-remaining_budget)*budget:.2f}/{budget:.2f} " +
              f"({(1-remaining_budget)*100:.1f}%)")
        
        # Track best performing run
        if total_reward > best_reward:
            best_reward = total_reward
            best_paths = env.added_paths.copy()
            best_state = env.state
            best_env = env  # Save the best environment for visualization
            print(f"  ✓ New best solution found!")
    
    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average Paths Added: {np.mean(all_num_paths):.1f} ± {np.std(all_num_paths):.1f}")
    print(f"Average Connectivity: {np.mean(all_connectivity):.4f} ± {np.std(all_connectivity):.4f}")
    print(f"Average Path Efficiency: {np.mean(all_path_efficiency):.4f} ± {np.std(all_path_efficiency):.4f}")
    print(f"Average Population Coverage: {np.mean(all_population):.4f} ± {np.std(all_population):.4f}")
    
    # Use the best performing run for visualization
    if best_paths and best_env:
        print("\nVisualizing best solution...")
        
        # Render the best solution using the saved best environment
        fig = best_env.render(mode='human')
        
        # Print detailed metrics for best solution
        conn, path, pop, budget_remain = best_state
        print(f"\nBest Solution Details:")
        print(f"Total Reward: {best_reward:.2f}")
        print(f"Connectivity: {conn:.4f}")
        print(f"Path Efficiency: {path:.4f}")
        print(f"Population Served: {pop:.4f}")
        print(f"Budget Used: {(1-budget_remain)*budget:.2f}/{budget:.2f} " +
              f"({(1-budget_remain)*100:.1f}%)")
        print(f"Added {len(best_paths)} bike path segments")
        
        # Calculate total length of added paths
        total_length = sum(length for _, _, length in best_paths)
        print(f"Total length of added bike paths: {total_length:.1f}m")
        
        # Export to shapefile if requested
        if export_shapefile:
            gdf = best_env.get_edge_gdf()
            if gdf is not None:
                if city_name:
                    safe_name = city_name.replace(', ', '_').replace(' ', '_')
                    output_file = f"suggested_bike_paths_{safe_name}.shp"
                    geojson_file = f"suggested_bike_paths_{safe_name}.geojson"
                else:
                    output_file = f"suggested_bike_paths_bounded_area.shp"
                    geojson_file = f"suggested_bike_paths_bounded_area.geojson"
                
                gdf.to_file(output_file)
                print(f"\nExported suggested bike paths to {output_file}")
                
                # Also save as GeoJSON for web visualization
                gdf.to_file(geojson_file, driver='GeoJSON')
                print(f"Exported suggested bike paths to {geojson_file}")
    else:
        print("\nNo valid solution found during evaluation.")
    
    return best_paths, best_state


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback for tracking and displaying training progress.
    """
    def __init__(self, verbose=1, log_freq=1000):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_connectivity = []
        self.episode_population = []
        self.start_time = None
        
    def _on_training_start(self):
        """Called at the start of training"""
        self.start_time = time.time()
        print("\nTraining started...")
        
    def _on_rollout_start(self):
        """Called at the start of a rollout"""
        pass
        
    def _on_step(self):
        """Called at each step"""
        # Check if an episode has ended
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                
                # Try to extract episode metrics from info
                if "episode" in infos[i]:
                    ep_info = infos[i]["episode"]
                    self.episode_rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])
                    
                # Try to extract custom metrics
                if "connectivity" in infos[i]:
                    self.episode_connectivity.append(infos[i]["connectivity"])
                if "population_served" in infos[i]:
                    self.episode_population.append(infos[i]["population_served"])
                
                # Log every few episodes
                if self.episode_count % 5 == 0:
                    elapsed_time = time.time() - self.start_time
                    avg_reward = sum(self.episode_rewards[-5:]) / 5
                    avg_length = sum(self.episode_lengths[-5:]) / 5
                    
                    print(f"\nEpisode {self.episode_count}: avg_reward={avg_reward:.2f}, avg_steps={avg_length:.1f}")
                    print(f"Progress: {self.num_timesteps}/{self.locals['total_timesteps']} steps " +
                          f"({self.num_timesteps/self.locals['total_timesteps']*100:.1f}%)")
                    print(f"Elapsed time: {elapsed_time/60:.2f} minutes")
        
        # Log every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            # Calculate average metrics
            if self.episode_rewards:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                print(f"\nStep {self.num_timesteps}: Running avg reward = {avg_reward:.2f} over {self.episode_count} episodes")
            
        return True
    
    def _on_training_end(self):
        """Called at the end of training"""
        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTraining completed after {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Total episodes: {self.episode_count}")
        
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            print(f"Average episode reward: {avg_reward:.2f}")
        
        print(f"Final timestep: {self.num_timesteps}")


def main():
    """Main function to demonstrate the bike path RL model."""
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate a RL model for bike path planning')
    parser.add_argument('--city', type=str, default='Colmar, France', 
                        help='City name to analyze (default: Amsterdam, Netherlands)')
    parser.add_argument('--budget', type=float, default=150000, 
                        help='Budget for bike path construction (default: 10000)')
    parser.add_argument('--timesteps', type=int, default=5000, 
                        help='Training timesteps (default: 10000)')
    parser.add_argument('--eval_episodes', type=int, default=3, 
                        help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--skip_training', action='store_true', 
                        help='Skip training and load existing model')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to existing model (for --skip_training)')
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Print header
    print("\n" + "="*70)
    print(f"BIKE PATH REINFORCEMENT LEARNING MODEL")
    print("="*70)
    print(f"City: {args.city}")
    print(f"Budget: {args.budget}")
    print(f"Training timesteps: {args.timesteps}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print("="*70 + "\n")
    
    # Skip training if requested
    if args.skip_training and args.model_path:
        from stable_baselines3 import PPO
        print(f"Loading existing model from {args.model_path}...")
        model = PPO.load(args.model_path)
    else:
        # Train the model
        model = train_rl_model(city_name=args.city, budget=args.budget, total_timesteps=args.timesteps)
    
    # Evaluate and visualize
    best_paths, best_state = evaluate_and_visualize(
        model, 
        city_name=args.city, 
        budget=args.budget,
        num_evaluations=args.eval_episodes
    )
    
    # Print total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return best_paths, best_state

if __name__ == "__main__":
    main()