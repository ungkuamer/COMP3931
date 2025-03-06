import osmnx as ox
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

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
        
        # 2. Average clustering coefficient (measure of local connectivity)
        clustering = np.mean(self.G_ig.transitivity_local_undirected(mode="zero"))
        
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
        centrality = self.G_ig.edge_betweenness()
        centrality_norm = (np.array(centrality) - min(centrality)) / (max(centrality) - min(centrality) + 1e-10)
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
        """
        # Check if action is valid (edge doesn't already have a bike path)
        if self.bike_paths[action]:
            # Invalid action - edge already has a bike path
            reward = -1.0  # Penalty for invalid action
            return self.state, reward, False, {"invalid_action": True}
        
        # Add bike path to selected edge
        self.bike_paths[action] = True
        self.G_ig.es[action]["bike_path"] = True
        
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
        
        return self.state, reward, done, {}
    
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
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Get bike path edges
            bike_edges = [e.tuple for e, has_path in zip(self.G_ig.es, self.bike_paths) if has_path]
            
            # Get node coordinates from original OSMnx graph
            node_coords = {}
            for node, data in self.G_osmnx.nodes(data=True):
                node_coords[node] = (data['x'], data['y'])
            
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
            
            plt.title(f"Bike Paths (Budget remaining: {self.remaining_budget})")
            plt.tight_layout()
            plt.show()
            
        return None

def get_city_network(city, distance=3000):
    """Get OSMnx network for a city with given distance from center"""
    # Download street network
    G = ox.graph_from_place(city, network_type='bike', simplify=True)
    
    # Project to UTM
    G = ox.project_graph(G)
    
    # Basic stats
    print(f"Network has {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    return G

def evaluate_bike_network(env, model=None):
    """Evaluate bike network quality with or without RL model"""
    if model:
        # Use trained model to suggest bike paths
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
    
    # Calculate metrics on final network
    final_state = env._get_state()
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

def main():
    # 1. Get city network
    city = "Amsterdam, Netherlands"  # Example city with good bike infrastructure
    G = get_city_network(city)
    
    # 2. Create environment
    env = BikePathEnvironment(G, budget=20)
    
    # 3. Set up RL model (PPO)
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=0.0003,
                gamma=0.99,
                n_steps=2048,
                ent_coef=0.01)
    
    # 4. Train model
    model.learn(total_timesteps=50000)
    
    # 5. Evaluate and visualize results
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}")
    
    # 6. Generate bike path recommendations using trained model
    obs = env.reset()
    done = False
    total_reward = 0
    
    # Step through environment using trained policy
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    
    print(f"Total reward: {total_reward}")
    
    # 7. Render final result
    env.render()
    
    # 8. Evaluate final bike network
    metrics = evaluate_bike_network(env)
    
    # 9. Compare with random baseline
    print("\nComparing with random baseline...")
    random_env = BikePathEnvironment(G, budget=20)
    
    # Random actions
    obs = random_env.reset()
    done = False
    while not done:
        action = random_env.action_space.sample()
        # Skip if invalid action (already has bike path)
        while random_env.bike_paths[action]:
            action = random_env.action_space.sample()
        obs, reward, done, _ = random_env.step(action)
    
    random_metrics = evaluate_bike_network(random_env)
    
    # Print comparison
    print("\nMetrics Comparison:")
    print(f"RL Model vs Random:")
    print(f"Coverage: {metrics['coverage']:.2f} vs {random_metrics['coverage']:.2f}")
    print(f"Connectivity: {metrics['connectivity']:.2f} vs {random_metrics['connectivity']:.2f}")
    print(f"Directness: {metrics['directness']:.2f} vs {random_metrics['directness']:.2f}")
    
if __name__ == "__main__":
    main()