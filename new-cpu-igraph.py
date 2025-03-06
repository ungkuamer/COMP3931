import osmnx as ox
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import argparse
import os
from datetime import datetime
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import multiprocessing as mp
from joblib import Parallel, delayed
import concurrent.futures
from functools import partial

def make_env(G_osmnx, budget=10, reward_weights=None, seed=0):
    """
    Create a function that returns a parameterized environment instance
    Used for vectorized environments
    """
    def _init():
        env = BikePathEnvironment(G_osmnx, budget=budget, reward_weights=reward_weights)
        env.seed(seed)
        return env
    return _init

class BikePathEnvironment(gym.Env):
    """Custom Environment for bike path planning using RL"""
    
    def __init__(self, G_osmnx, budget=10, reward_weights=None):
        super(BikePathEnvironment, self).__init__()
        
        # Convert OSMnx graph to igraph
        self.G_osmnx = G_osmnx
        self.G_ig = self._convert_to_igraph(G_osmnx)
        
        # Store original graph for reset
        self.original_G_ig = self.G_ig.copy()
        
        # Budget = number of edges that can be added
        self.budget = budget
        self.remaining_budget = budget
        
        # Set reward weights for multi-objective optimization
        self.reward_weights = reward_weights or {
            "connectivity": 0.4,
            "directness": 0.3,
            "coverage": 0.3
        }
        
        # Define action space: choose an edge to add a bike path
        # Each action is an edge index
        self.action_space = spaces.Discrete(self.G_ig.ecount())
        
        # Define observation space: features of the current graph state
        # Including bike path coverage, connectivity metrics, etc.
        num_features = 5  # Number of graph-level features we're tracking
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_features,), dtype=np.float32
        )
        
        # Track which edges already have bike paths
        self.bike_paths = np.zeros(self.G_ig.ecount(), dtype=bool)
        
        # Initialize state
        self.state = self._get_state()
    
    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def _convert_to_igraph(self, G_osmnx):
        """Convert OSMnx graph to igraph format"""
        # Extract edges with their attributes
        edges = []
        edge_attrs = {}
        
        for u, v, k, data in G_osmnx.edges(keys=True, data=True):
            edges.append((u, v))
            for key, value in data.items():
                if key not in edge_attrs:
                    edge_attrs[key] = []
                while len(edge_attrs[key]) < len(edges) - 1:
                    edge_attrs[key].append(None)
                edge_attrs[key].append(value)
        
        # Create igraph from edge list
        G_ig = ig.Graph.TupleList(edges, directed=G_osmnx.is_directed())
        
        # Add edge attributes
        for key, values in edge_attrs.items():
            if len(values) < G_ig.ecount():
                values.extend([None] * (G_ig.ecount() - len(values)))
            G_ig.es[key] = values
        
        # Add node attributes
        for node, data in G_osmnx.nodes(data=True):
            if node in G_ig.vs["name"]:
                node_idx = G_ig.vs.find(name=node).index
                for key, value in data.items():
                    G_ig.vs[node_idx][key] = value
        
        # Add bike path attribute to all edges (initially False)
        G_ig.es["bike_path"] = False
        
        return G_ig
    
    def _get_state(self):
        """Get current state representation of the graph"""
        # Calculate network metrics relevant to bike paths
        
        # 1. Bike path coverage (percentage of edges with bike paths)
        coverage = np.mean(self.bike_paths)
        
        # Use parallel algorithms where possible in igraph
        
        # 2. Average clustering coefficient (measure of local connectivity)
        # Set parallel for multi-threading in transitivity calculation (n-1 cores)
        num_parallel = max(1, mp.cpu_count() - 1)
        clustering = np.mean(self.G_ig.transitivity_local_undirected(mode="zero", parallel=num_parallel))
        
        # 3. Bike path connectivity (largest connected component of bike paths)
        # Create subgraph with only bike path edges
        bike_subgraph = self.G_ig.subgraph_edges(
            self.G_ig.es.select(bike_path_eq=True), delete_vertices=False
        )
        components = bike_subgraph.components()
        if components.giant().vcount() > 0:
            connectivity = components.giant().vcount() / self.G_ig.vcount()
        else:
            connectivity = 0
        
        # 4. Centrality distribution (how central are the bike paths)
        # Enable parallelism in betweenness calculation (n-1 cores)
        num_parallel = max(1, mp.cpu_count() - 1)
        centrality = self.G_ig.edge_betweenness(parallel=num_parallel)
        
        # Use numpy vectorized operations for better performance
        min_centrality = np.min(centrality) if centrality else 0
        max_centrality = np.max(centrality) if centrality else 1
        range_centrality = max_centrality - min_centrality + 1e-10
        
        # Vectorized normalization is faster than element-wise
        centrality_norm = (np.array(centrality) - min_centrality) / range_centrality
        
        # Use numpy's masked operations for conditional means
        bike_path_centrality = np.mean(centrality_norm[self.bike_paths]) if np.any(self.bike_paths) else 0
        
        # 5. Directness (average path length improvement for bike riders)
        # For simplicity, we use a proxy: proportion of high-betweenness edges with bike paths
        high_betweenness = centrality_norm > 0.7
        directness = np.mean(self.bike_paths[high_betweenness]) if np.any(high_betweenness) else 0
        
        return np.array([coverage, clustering, connectivity, bike_path_centrality, directness], dtype=np.float32)
    
    def step(self, action):
        """
        Take action to add a bike path and return next state, reward, done, info
        Action: index of edge to add bike path
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Ensure action is within bounds
        action = int(action)  # Convert to int if it's a numpy int
        if action < 0 or action >= self.G_ig.ecount():
            reward = -1.0  # Penalty for invalid action
            return self.state, reward, False, {"invalid_action": True, "message": "Action out of bounds"}
            
        # Check if action is valid (edge doesn't already have a bike path)
        if self.bike_paths[action]:
            # Invalid action - edge already has a bike path
            reward = -1.0  # Penalty for invalid action
            return self.state, reward, False, {"invalid_action": True, "message": "Edge already has a bike path"}
        
        # Add bike path to selected edge
        self.bike_paths[action] = True
        self.G_ig.es[action]["bike_path"] = True
        
        # Get edge details for info
        edge = self.G_ig.es[action]
        source_node = self.G_ig.vs[edge.source]["name"]
        target_node = self.G_ig.vs[edge.target]["name"]
        
        # Update state
        new_state = self._get_state()
        
        # Calculate reward based on improvement in state features
        reward = self._calculate_reward(self.state, new_state)
        
        # Update state
        self.state = new_state
        
        # Decrease remaining budget
        self.remaining_budget -= 1
        
        # Check if episode is done (budget exhausted)
        done = self.remaining_budget <= 0
        
        # Return state, reward, done, and info
        return self.state, reward, done, {
            "action": action,
            "source_node": source_node,
            "target_node": target_node,
            "reward": reward,
            "remaining_budget": self.remaining_budget
        }
    
    def _calculate_reward(self, old_state, new_state):
        """Calculate reward based on improvement in state metrics"""
        # Extract metrics from states
        old_coverage, old_clustering, old_connectivity, old_centrality, old_directness = old_state
        new_coverage, new_clustering, new_connectivity, new_centrality, new_directness = new_state
        
        # Calculate improvements
        coverage_improvement = new_coverage - old_coverage
        connectivity_improvement = new_connectivity - old_connectivity
        directness_improvement = new_directness - old_directness
        
        # Calculate weighted reward
        reward = (
            self.reward_weights["connectivity"] * connectivity_improvement +
            self.reward_weights["directness"] * directness_improvement +
            self.reward_weights["coverage"] * coverage_improvement
        )
        
        # Scale reward
        reward = reward * 10
        
        return reward
    
    def reset(self):
        """Reset environment to initial state"""
        # Reset graph to original state
        self.G_ig = self.original_G_ig.copy()
        
        # Reset bike paths
        self.bike_paths = np.zeros(self.G_ig.ecount(), dtype=bool)
        self.G_ig.es["bike_path"] = False
        
        # Reset budget
        self.remaining_budget = self.budget
        
        # Reset state
        self.state = self._get_state()
        
        return self.state
    
    def render(self, mode='human', filename=None, use_parallel=True, num_workers=max(1, mp.cpu_count() - 1)):
        """Render the environment with optional parallelization

        Args:
            mode (str): Rendering mode
            filename (str, optional): If provided, save figure to this file
            use_parallel (bool): Whether to use parallel processing for rendering
            num_workers (int, optional): Number of worker processes
        """
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get bike path edges
        bike_edges = [e.tuple for e, has_path in zip(self.G_ig.es, self.bike_paths) if has_path]
        
        # Get node coordinates from original OSMnx graph
        node_coords = {}
        for node, data in self.G_osmnx.nodes(data=True):
            node_coords[node] = (data['x'], data['y'])
        
        if use_parallel and len(self.G_ig.es) > 1000:  # Only use parallel for large graphs
            # Determine number of workers
            # Use provided num_workers (default is mp.cpu_count() - 1)
            
            # Parallel processing for original graph edges
            # Split edges into chunks for parallel processing
            all_edges = list(self.G_ig.es)
            chunk_size = max(1, len(all_edges) // num_workers)
            edge_chunks = [all_edges[i:i+chunk_size] for i in range(0, len(all_edges), chunk_size)]
            
            # Function to process a chunk of edges
            def process_edge_chunk(edges):
                lines = []
                for edge in edges:
                    u, v = self.G_ig.vs[edge.source]["name"], self.G_ig.vs[edge.target]["name"]
                    if u in node_coords and v in node_coords:
                        x1, y1 = node_coords[u]
                        x2, y2 = node_coords[v]
                        lines.append(((x1, x2), (y1, y2)))
                return lines
            
            # Process regular edges in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_edge_chunk, edge_chunks))
            
            # Plot all regular edges
            for chunk_lines in results:
                for (x1, x2), (y1, y2) in chunk_lines:
                    ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.3, linewidth=1)
            
            # Process bike path edges (typically fewer, so less benefit from parallelization)
            bike_lines = []
            for u, v in bike_edges:
                u_name, v_name = self.G_ig.vs[u]["name"], self.G_ig.vs[v]["name"]
                if u_name in node_coords and v_name in node_coords:
                    x1, y1 = node_coords[u_name]
                    x2, y2 = node_coords[v_name]
                    bike_lines.append(((x1, x2), (y1, y2)))
            
            # Plot bike path edges
            for (x1, x2), (y1, y2) in bike_lines:
                ax.plot([x1, x2], [y1, y2], color='green', linewidth=2)
        else:
            # Sequential rendering for smaller graphs
            # Plot original graph edges
            for edge in self.G_ig.es:
                u, v = self.G_ig.vs[edge.source]["name"], self.G_ig.vs[edge.target]["name"]
                if u in node_coords and v in node_coords:
                    x1, y1 = node_coords[u]
                    x2, y2 = node_coords[v]
                    ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.3, linewidth=1)
            
            # Plot bike path edges
            for u, v in bike_edges:
                u_name, v_name = self.G_ig.vs[u]["name"], self.G_ig.vs[v]["name"]
                if u_name in node_coords and v_name in node_coords:
                    x1, y1 = node_coords[u_name]
                    x2, y2 = node_coords[v_name]
                    ax.plot([x1, x2], [y1, y2], color='green', linewidth=2)
        
        # Add a legend
        ax.plot([], [], color='green', linewidth=2, label='Proposed Bike Paths')
        ax.plot([], [], color='gray', alpha=0.3, linewidth=1, label='Road Network')
        ax.legend(loc='upper right')
        
        plt.title(f"Bike Paths (Budget remaining: {self.remaining_budget})")
        plt.tight_layout()
        
        if filename:
            # Save figure to file
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {filename}")
        
        if mode == 'human':
            plt.show()
        
        plt.close(fig)
        return None

# Ray Remote function for distributed graph processing was removed

def get_city_network(city, distance=3000, retry_with_drive=True):
    """Get OSMnx network for a city with given distance from center
    
    Args:
        city (str): Name of the city
        distance (int): Distance in meters from center
        retry_with_drive (bool): If True, retry with drive network if bike network fails
        
    Returns:
        nx.MultiDiGraph: OSMnx graph
    """
    try:
        print(f"Downloading bike network for {city}...")
        # Download street network
        G = ox.graph_from_place(city, network_type='bike', simplify=True)
    except Exception as e:
        print(f"Error downloading bike network: {e}")
        if retry_with_drive:
            print("Retrying with drive network...")
            try:
                G = ox.graph_from_place(city, network_type='drive', simplify=True)
                print("Successfully downloaded drive network.")
            except Exception as e2:
                print(f"Error downloading drive network: {e2}")
                print("Trying with smaller bounding box around city center...")
                try:
                    # Get city center coordinates
                    geocode = ox.geocoder.geocode(city)
                    # Create a smaller bounding box
                    G = ox.graph_from_address(city, dist=distance, network_type='drive', simplify=True)
                    print(f"Successfully downloaded network for a {distance}m radius around city center.")
                except Exception as e3:
                    raise Exception(f"Failed to download network for {city}: {e3}")
        else:
            raise e
    
    # Project to UTM
    G = ox.project_graph(G)
    
    # Basic stats
    print(f"Network has {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Check if the network is too large
    if len(G.edges) > 10000:
        print("Network is very large. Consider using a smaller area or increasing computational resources.")
    
    return G

def parallel_evaluate_model(env, model, n_episodes=3, n_jobs=-1):
    """Parallel evaluation of model on multiple episodes
    
    Args:
        env (BikePathEnvironment): Environment to evaluate
        model: Model to evaluate
        n_episodes (int): Number of episodes
        n_jobs (int): Number of parallel jobs (-1 for all cores)
        
    Returns:
        list: Rewards for each episode
    """
    # Create a partial function for evaluation
    def evaluate_episode(ep):
        env_copy = BikePathEnvironment(
            env.G_osmnx, 
            budget=env.budget, 
            reward_weights=env.reward_weights
        )
        obs = env_copy.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env_copy.step(action)
            total_reward += reward
        return total_reward
    
    # Use joblib to parallelize evaluation
    rewards = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_episode)(ep) for ep in range(n_episodes)
    )
    
    return rewards

def evaluate_bike_network(env, model=None, progress_bar=True, parallel=True):
    """Evaluate bike network quality with or without RL model
    
    Args:
        env (BikePathEnvironment): Environment to evaluate
        model (stable_baselines3.PPO, optional): Model to use for suggesting bike paths
        progress_bar (bool): Whether to show progress bar during evaluation
        parallel (bool): Whether to use parallel processing
        
    Returns:
        dict: Metrics including coverage, connectivity, and directness
    """
    if model:
        if parallel:
            # Use parallel evaluation
            env_copy = BikePathEnvironment(
                env.G_osmnx, 
                budget=env.budget, 
                reward_weights=env.reward_weights
            )
            obs = env_copy.reset()
            done = False
            
            if progress_bar:
                pbar = tqdm(total=env_copy.budget, desc="Evaluating with model")
            
            # Process actions in batches for better parallelism
            while not done:
                # Get next action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step in environment
                prev_obs = obs
                obs, reward, done, info = env_copy.step(action)
                
                # Only update progress if action was valid
                if not info.get('invalid_action', False) and progress_bar:
                    pbar.update(1)
                    pbar.set_postfix({"reward": f"{reward:.2f}"})
            
            if progress_bar:
                pbar.close()
                
            # Calculate metrics on final network
            final_state = env_copy._get_state()
        else:
            # Use regular evaluation
            obs = env.reset()
            done = False
            steps = 0
            budget = env.budget
            
            if progress_bar:
                pbar = tqdm(total=budget, desc="Evaluating with model")
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                prev_obs = obs
                obs, reward, done, info = env.step(action)
                
                # Only update progress if action was valid
                if not info.get('invalid_action', False):
                    steps += 1
                    if progress_bar:
                        pbar.update(1)
                        pbar.set_postfix({"reward": f"{reward:.2f}"})
            
            if progress_bar:
                pbar.close()
            
            # Calculate metrics on final network
            final_state = env._get_state()
    else:
        # Just get current state if no model is provided
        final_state = env._get_state()
    
    # Extract metrics
    coverage = final_state[0]
    connectivity = final_state[2]
    directness = final_state[4]
    
    print(f"Bike Network Metrics:")
    print(f"Coverage: {coverage:.2f}")
    print(f"Connectivity: {connectivity:.2f}")
    print(f"Directness: {directness:.2f}")
    
    return {
        "coverage": coverage,
        "connectivity": connectivity,
        "directness": directness
    }

def save_bike_network_to_geojson(env, filename):
    """Save the bike network to a GeoJSON file
    
    Args:
        env (BikePathEnvironment): Environment with bike paths
        filename (str): Output filename
    """
    import geopandas as gpd
    from shapely.geometry import LineString
    
    # Get bike path edges
    bike_edges = [e for e, has_path in zip(env.G_ig.es, env.bike_paths) if has_path]
    
    # Get node coordinates from original OSMnx graph
    node_coords = {}
    for node, data in env.G_osmnx.nodes(data=True):
        node_coords[node] = (data['x'], data['y'])
    
    # Function to process a single edge
    def process_edge(edge):
        u, v = env.G_ig.vs[edge.source]["name"], env.G_ig.vs[edge.target]["name"]
        if u in node_coords and v in node_coords:
            x1, y1 = node_coords[u]
            x2, y2 = node_coords[v]
            return LineString([(x1, y1), (x2, y2)])
        return None
    
    # Parallel processing of edges
    with concurrent.futures.ProcessPoolExecutor() as executor:
        geometries = list(filter(None, executor.map(process_edge, bike_edges)))
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=geometries)
    
    # Save to GeoJSON
    gdf.to_file(filename, driver='GeoJSON')
    print(f"Bike network saved to {filename}")

def create_interactive_map(env, filename):
    """Create an interactive HTML map of the bike network
    
    Args:
        env (BikePathEnvironment): Environment with bike paths
        filename (str): Output HTML filename
    """
    import folium
    
    # Get center of the network
    center_node = list(env.G_osmnx.nodes())[0]
    center_lat = env.G_osmnx.nodes[center_node]['y']
    center_lon = env.G_osmnx.nodes[center_node]['x']
    
    # Create a map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Get bike path edges
    bike_edges = [e for e, has_path in zip(env.G_ig.es, env.bike_paths) if has_path]
    
    # Get node coordinates from original OSMnx graph
    node_coords = {}
    for node, data in env.G_osmnx.nodes(data=True):
        node_coords[node] = (data['y'], data['x'])  # Note lat/lon order
    
    # Function to process an edge batch for parallel execution
    def process_edge_batch(edges_batch):
        polylines = []
        for edge in edges_batch:
            u, v = env.G_ig.vs[edge.source]["name"], env.G_ig.vs[edge.target]["name"]
            if u in node_coords and v in node_coords:
                polylines.append((node_coords[u], node_coords[v]))
        return polylines
    
    # Split edges into batches for parallel processing
    batch_size = max(1, len(bike_edges) // mp.cpu_count())
    edge_batches = [bike_edges[i:i+batch_size] for i in range(0, len(bike_edges), batch_size)]
    
    # Process edge batches in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        all_polylines = list(executor.map(process_edge_batch, edge_batches))
    
    # Flatten the list of polylines
    polylines = [item for sublist in all_polylines for item in sublist]
    
    # Add polylines to map
    for coords in polylines:
        folium.PolyLine(
            coords,
            color='green',
            weight=4,
            opacity=0.8,
            tooltip='Proposed Bike Path'
        ).add_to(m)
    
    # Save the map
    m.save(filename)
    print(f"Interactive map saved to {filename}")

class ProgressCallback(BaseCallback):
    """
    Custom callback for tracking training progress
    """
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.progress_bar = None
        
    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals.get('total_timesteps'), 
                                desc="Training Progress", 
                                unit="steps")
        
    def _on_step(self):
        self.progress_bar.update(1)
        return True
        
    def _on_training_end(self):
        self.progress_bar.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bike Path Planning with Reinforcement Learning")
    
    # City configuration
    parser.add_argument("--city", type=str, default="Colmar, France",
                        help="City name for analysis (e.g., 'Colmar, France')")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="Total timesteps for training")
    parser.add_argument("--eval_episodes", type=int, default=3,
                        help="Number of episodes for evaluation")
    
    # Environment parameters
    parser.add_argument("--budget", type=int, default=150,
                        help="Budget (number of bike paths to add)")
    
    # Reward weights
    parser.add_argument("--connectivity_weight", type=float, default=0.4,
                        help="Weight for connectivity in reward function")
    parser.add_argument("--directness_weight", type=float, default=0.3,
                        help="Weight for directness in reward function")
    parser.add_argument("--coverage_weight", type=float, default=0.3,
                        help="Weight for coverage in reward function")
    
    # Save options
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory to save trained models")
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="Frequency of saving model checkpoints (in timesteps)")
    
    # Parallelization options
    parser.add_argument("--num_envs", type=int, default=max(1, mp.cpu_count() - 1),
                        help="Number of parallel environments for training")
    parser.add_argument("--num_eval_processes", type=int, default=max(1, mp.cpu_count() - 1),
                        help="Number of parallel processes for evaluation")
    
    # Visualization options
    parser.add_argument("--render", action="store_true",
                        help="Render final result")
    parser.add_argument("--interactive_map", action="store_true",
                        help="Create interactive map")
    parser.add_argument("--export_geojson", action="store_true",
                        help="Export bike paths as GeoJSON")
    
    return parser.parse_args()

def main():
    # 0. Parse arguments
    args = parse_arguments()
    
    # Automatically adjust num_envs and num_eval_processes if not explicitly set
    if args.num_envs == max(1, mp.cpu_count() - 1):  # If using the default value
        recommended_cores = max(1, mp.cpu_count() - 1)
        args.num_envs = recommended_cores
        print(f"Automatically using {recommended_cores} cores for environments (keeping 1 core free)")
    
    if args.num_eval_processes == max(1, mp.cpu_count() - 1):  # If using the default value
        recommended_cores = max(1, mp.cpu_count() - 1)
        args.num_eval_processes = recommended_cores
        print(f"Automatically using {recommended_cores} cores for evaluation (keeping 1 core free)")
    
    # Print system info for parallelization
    print(f"System information:")
    print(f"  CPU cores available: {mp.cpu_count()}")
    print(f"  Cores used for environments: {args.num_envs}")
    print(f"  Cores used for evaluation: {args.num_eval_processes}")
    print(f"  Python multiprocessing start method: {mp.get_start_method(allow_none=True)}")
    
    # Set environment variables for better parallelization
    os.environ["OMP_NUM_THREADS"] = str(1)  # Limit OpenMP threads
    os.environ["MKL_NUM_THREADS"] = str(1)  # Limit MKL threads
    os.environ["NUMEXPR_NUM_THREADS"] = str(1)  # Limit numexpr threads
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_name = args.city.split(',')[0].replace(' ', '_')
    output_dir = os.path.join("outputs", f"{city_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    
    print(f"Configuration:")
    print(f"  City: {args.city}")
    print(f"  Budget: {args.budget}")
    print(f"  Training timesteps: {args.timesteps}")
    print(f"  Evaluation episodes: {args.eval_episodes}")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Evaluation processes: {args.num_eval_processes}")
    
    # 1. Get city network
    print(f"Downloading street network for {args.city}...")
    G = get_city_network(args.city)
    
    # Create reward weights dictionary
    reward_weights = {
        "connectivity": args.connectivity_weight,
        "directness": args.directness_weight,
        "coverage": args.coverage_weight
    }
    
    # 2. Create vectorized environment for parallel training
    print(f"Creating {args.num_envs} parallel environments with budget of {args.budget} bike paths...")
    
    # Create environment creation functions
    env_fns = [make_env(G, budget=args.budget, reward_weights=reward_weights, seed=i) 
              for i in range(args.num_envs)]
    
    # Create vectorized environment
    train_env = SubprocVecEnv(env_fns)
    
    # Create a single environment for evaluation and visualization
    eval_env = BikePathEnvironment(G, budget=args.budget, reward_weights=reward_weights)
    
    # Set up model directory
    model_dir = os.path.join(args.model_dir, f"{args.city.split(',')[0]}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up callbacks
    progress_callback = ProgressCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,  # Adjust save frequency for vectorized environment
        save_path=model_dir,
        name_prefix="bike_path_model"
    )
    
    # 3. Set up RL model with parallelized training
    print("Initializing PPO model with parallel environments...")
    model = PPO("MlpPolicy", train_env, verbose=1, 
                learning_rate=0.0003,
                gamma=0.99,
                n_steps=2048 // args.num_envs,  # Adjust steps for vectorized env
                ent_coef=0.01,
                n_epochs=10,  # Increase training epochs
                device="cpu")
    
    # 4. Train model
    print(f"Starting training for {args.timesteps} timesteps using {args.num_envs} parallel environments...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[progress_callback, checkpoint_callback]
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # 5. Evaluate model in parallel
    print(f"Evaluating model over {args.eval_episodes} episodes using {args.num_eval_processes} processes...")
    
    # Joblib-based parallel evaluation
    rewards = parallel_evaluate_model(
        eval_env, model, 
        n_episodes=args.eval_episodes, 
        n_jobs=args.num_eval_processes
    )
    mean_reward = np.mean(rewards)
    
    print(f"Mean reward: {mean_reward:.2f}")
    
if __name__ == "__main__":
    # Add additional visualization arguments
    parser = argparse.ArgumentParser(description="Bike Path Planning with Reinforcement Learning")
    parser.add_argument("--city", type=str, default="Colmar, France",
                        help="City name for analysis (e.g., 'Colmar, France')")
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="Total timesteps for training")
    parser.add_argument("--eval_episodes", type=int, default=3,
                        help="Number of episodes for evaluation")
    parser.add_argument("--budget", type=int, default=100,
                        help="Budget (number of bike paths to add)")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory to save trained models")
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="Frequency of saving model checkpoints (in timesteps)")
    parser.add_argument("--render", action="store_true",
                        help="Render final result")
    parser.add_argument("--interactive_map", action="store_true",
                        help="Create interactive map")
    parser.add_argument("--export_geojson", action="store_true",
                        help="Export bike paths as GeoJSON")
    
    args = parser.parse_args()
    main()