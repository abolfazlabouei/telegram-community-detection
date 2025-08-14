from html import parser
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import argparse
import pandas as pd
import random
from networkx.algorithms.approximation import diameter

class GraphAnalyzer:
    def __init__(self, file_path, file_type='edgelist'):
        self.graph = self._load_graph(file_path, file_type)
        if self.graph:
            print("Graph loaded successfully.")
            self.nodes = list(self.graph.nodes())

    def _load_graph(self, file_path, file_type):
        try:
            if file_type == 'pkl':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif file_type == 'edgelist':
                def read_edgelist_with_attrs(path):
                    G = nx.Graph()
                    df = pd.read_csv(path, header=None)
                    for _, row in df.iterrows():
                        u, v, data = row[0], row[1], row[2]
                        try:
                            attr = eval(data)
                        except:
                            attr = {}
                        G.add_edge(u, v, **attr)
                    return G
                return read_edgelist_with_attrs(file_path)
            else:
                print(f"Error: Unsupported file type '{file_type}'.")
                return None
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error loading graph: {e}")
            return None

    def display_basic_info(self):
        print("\n--- Graph Information ---")
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print("-------------------------\n")

    def get_structural_metrics(self):
        print("\n--- Structural Metrics ---")
        print(f"Density: {nx.density(self.graph):.6f}")
        print(f"Connected Components: {nx.number_connected_components(self.graph)}")
        largest_cc = max(nx.connected_components(self.graph), key=len)
        Gcc = self.graph.subgraph(largest_cc).copy()
        print(f"Largest Connected Component Size: {Gcc.number_of_nodes()}")
        try:
            print(f"Approx. Diameter (LCC): {diameter(Gcc)}")
        except:
            print("Could not compute approximate diameter.")
        print("----------------------------\n")

    def get_degree_centrality(self, n=5):
        print(f"\n--- Top {n} Nodes by Degree Centrality ---")
        dc = nx.degree_centrality(self.graph)
        top = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:n]
        for i, (node, val) in enumerate(top):
            print(f"{i+1}. Node {node}: {val:.4f}")
        print("------------------------------------------\n")

    def get_betweenness_centrality(self, n=5, k=1000):
        print(f"\n--- Top {n} Nodes by Betweenness Centrality (k={k}) ---")
        k = min(k, self.graph.number_of_nodes())
        bc = nx.betweenness_centrality(self.graph, k=k, seed=42)
        top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:n]
        for i, (node, val) in enumerate(top):
            print(f"{i+1}. Node {node}: {val:.4f}")
        print("---------------------------------------------------------------------\n")

    def get_average_clustering(self):
        print("\n--- Graph Clustering ---")
        avg_clust = nx.average_clustering(self.graph)
        print(f"Average clustering coefficient: {avg_clust:.4f}")
        print("------------------------\n")

    def draw_subgraph(self, size=100, save_path=None):
        if size > self.graph.number_of_nodes():
            sub_nodes = self.nodes
        else:
            sub_nodes = random.sample(self.nodes, size)
        subgraph = self.graph.subgraph(sub_nodes)
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw(subgraph, pos, with_labels=True, node_size=50, font_size=8,
                node_color='skyblue', edge_color='gray', width=0.5)
        plt.title(f"Subgraph of {size} nodes")
        if save_path:
            plt.savefig(save_path)
            print(f"📸 Subgraph saved to {save_path}")
        else:
            plt.show()

    def draw_top_degree_nodes(self, top_n=100, save_path=None):
        degrees = dict(self.graph.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]
        subgraph = self.graph.subgraph(top_nodes)
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw(subgraph, pos, with_labels=True, node_size=100, font_size=8,
                node_color='orange', edge_color='gray', width=0.8)
        plt.title(f"Top {top_n} nodes by degree")
        if save_path:
            plt.savefig(save_path)
            print(f"📸 Top-degree subgraph saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph Analysis Tool")
    parser.add_argument("filepath", help="Path to the graph file.")
    parser.add_argument("--type", choices=['edgelist', 'pkl'], default='pkl',
                        help="Type of the graph file (default: pkl).")
    parser.add_argument("--info", action="store_true", help="Display basic graph info.")
    parser.add_argument("--structure", action="store_true", help="Show structural metrics.")
    parser.add_argument("--degree", type=int, nargs='?', const=5, metavar='N',
                        help="Top N nodes by degree centrality (default: 5).")
    parser.add_argument("--betweenness", type=int, nargs='?', const=5, metavar='N',
                        help="Top N nodes by betweenness centrality (default: 5).")
    parser.add_argument("--k_samples", type=int, default=1000,
                        help="Number of samples for betweenness approximation (default: 1000).")
    parser.add_argument("--clustering", action="store_true", help="Calculate average clustering.")
    parser.add_argument("--draw", type=int, nargs='?', const=100, metavar='SIZE',
                        help="Draw a random subgraph (default: 100 nodes).")
    parser.add_argument("--draw_degree", type=int, nargs='?', const=100, metavar='N',
                        help="Draw top-N degree subgraph (default: 100).")
    parser.add_argument("--draw_save", type=str,
                        help="Save path for drawn graph images (e.g., out.png).")

    args = parser.parse_args()

    # Create analyzer object
    analyzer = GraphAnalyzer(args.filepath, args.type)

    if analyzer.graph:
        if args.info:
            analyzer.display_basic_info()
        if args.structure:
            analyzer.get_structural_metrics()
        if args.degree is not None:
            analyzer.get_degree_centrality(args.degree)
        if args.betweenness is not None:
            analyzer.get_betweenness_centrality(args.betweenness, args.k_samples)
        if args.clustering:
            analyzer.get_average_clustering()
        if args.draw is not None:
            analyzer.draw_subgraph(args.draw, save_path=args.draw_save)
        if args.draw_degree is not None:
            analyzer.draw_top_degree_nodes(args.draw_degree, save_path=args.draw_save)

        # پیش‌فرض اگر هیچ کاری درخواست نشد، فقط info رو چاپ کنه
        if not any([args.info, args.structure, args.degree, args.betweenness, args.clustering, args.draw, args.draw_degree]):
            analyzer.display_basic_info()

