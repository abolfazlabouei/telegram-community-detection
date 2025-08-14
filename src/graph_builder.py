import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
import pickle

class GraphBuilder:
    def __init__(self, membership_path: str, min_common_members: int = 5):
        self.membership_path = membership_path
        self.min_common_members = min_common_members
        self.membership_df = None
        self.user_encoder = LabelEncoder()
        self.group_encoder = LabelEncoder()
        self.membership_matrix = None
        self.group_overlap = None
        self.graph = nx.Graph()
        self.index_to_groupID = None

    def load_membership(self):
        self.membership_df = pd.read_csv(self.membership_path)
        print(f"✅ Loaded {len(self.membership_df)} membership records.")

    def build_sparse_matrix(self):
        group_ids = self.group_encoder.fit_transform(self.membership_df["groupID"])
        user_ids = self.user_encoder.fit_transform(self.membership_df["userID"])
        data = [1] * len(self.membership_df)
        self.membership_matrix = coo_matrix((data, (user_ids, group_ids)))
        print(f"✅ Created sparse membership matrix with shape {self.membership_matrix.shape}.")

    def compute_group_overlap(self):
        self.group_overlap = self.membership_matrix.T @ self.membership_matrix
        self.group_overlap.setdiag(0)  # remove self-loops
        print("✅ Computed group overlap matrix.")

    def extract_edges(self):
        coo = self.group_overlap.tocoo()
        mask = coo.data >= self.min_common_members
        rows = coo.row[mask]
        cols = coo.col[mask]
        weights = coo.data[mask]

        self.index_to_groupID = self.group_encoder.inverse_transform(
            np.arange(len(self.group_encoder.classes_))
        )

        edges = [
            (int(self.index_to_groupID[i]), int(self.index_to_groupID[j]), int(w))
            for i, j, w in zip(rows, cols, weights)
            if i < j
        ]
        print(f"✅ Extracted {len(edges)} edges with ≥ {self.min_common_members} shared members.")
        return edges

    def build_graph(self):
        edges = self.extract_edges()
        self.graph.add_weighted_edges_from(edges)
        print(f"📌 Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def save_graph_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"✅ Graph saved to {path} (pickle format).")

    def save_edge_list(self, path: str):
        nx.write_weighted_edgelist(self.graph, path)
        print(f"✅ Graph edge list saved to {path}.")

if __name__=="__main__":
    builder = GraphBuilder(
    membership_path="data/processed/membership_filtered.csv",
    min_common_members=5
)

    builder.load_membership()
    builder.build_sparse_matrix()
    builder.compute_group_overlap()
    builder.build_graph()


    builder.save_graph_pickle("results/telegram_graph.pkl")
    builder.save_edge_list("results/telegram_graph.edgelist")