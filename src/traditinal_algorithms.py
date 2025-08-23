import pandas as pd
import pickle
import networkx as nx
from pathlib import Path
import logging
import time
from typing import Dict, Tuple, Any, Optional, List
import warnings

# Community detection algorithms
import community.community_louvain as community_louvain
import igraph as ig
from infomap import Infomap


class CommunityDetection:
    def __init__(self, graph_path: str, group_info_path: str, results_dir: str = "results"):
        self.graph_path = graph_path
        self.group_info_path = group_info_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Initialize data containers
        self.G_nx: Optional[nx.Graph] = None
        self.G_ig: Optional[ig.Graph] = None
        self.df_info: Optional[pd.DataFrame] = None

        # Node index mappings
        self.node_to_idx: Dict[Any, int] = {}
        self.idx_to_node: Dict[int, Any] = {}

        # Setup logging
        self._setup_logging()

        # Store all results: algo_name -> (partition_dict, modularity)
        self.all_results: Dict[str, Tuple[Dict[Any, Any], float]] = {}

    def _setup_logging(self) -> None:
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.results_dir / "community_detection.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> None:
        """Load and prepare data with enhanced error handling."""
        start_time = time.time()
        self.logger.info("🔄 Loading data...")

        try:
            # Load NetworkX graph
            with open(self.graph_path, "rb") as f:
                self.G_nx = pickle.load(f)
            if not isinstance(self.G_nx, (nx.Graph, nx.DiGraph)):
                raise TypeError("Loaded object is not a NetworkX Graph/DiGraph.")

            # Ensure undirected for modularity comparability
            if isinstance(self.G_nx, nx.DiGraph):
                self.logger.info("Converting DiGraph to undirected Graph for community detection.")
                self.G_nx = self.G_nx.to_undirected()

            # Load group info
            self.df_info = pd.read_csv(self.group_info_path)
            original_count = len(self.df_info)

            # Filter nodes that exist in graph
            if "peerid" not in self.df_info.columns:
                raise KeyError("CSV must contain a 'peerid' column.")
            self.df_info = self.df_info[self.df_info["peerid"].isin(self.G_nx.nodes)]
            filtered_count = len(self.df_info)

            # Convert to iGraph with proper node mapping
            self._create_igraph()

            load_time = time.time() - start_time
            self.logger.info(f"✅ Data loaded in {load_time:.2f}s")
            self.logger.info(f"📊 Graph: {self.G_nx.number_of_nodes()} nodes, {self.G_nx.number_of_edges()} edges")
            self.logger.info(f"📊 Groups: {original_count} → {filtered_count} (filtered)")

        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            raise

    def _create_igraph(self) -> None:
        """Create iGraph with proper node mapping."""
        try:
            node_list = list(self.G_nx.nodes())
            self.node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

            # Edges as indices
            edges_idx = [(self.node_to_idx[u], self.node_to_idx[v]) for u, v in self.G_nx.edges()]
            weights = [self.G_nx[u][v].get("weight", 1.0) for u, v in self.G_nx.edges()]

            # Build igraph
            self.G_ig = ig.Graph(n=len(node_list), edges=edges_idx)
            self.G_ig.vs["name"] = node_list
            # Assign weights only if any non-default present or any edges exist
            if len(weights) == self.G_ig.ecount() and (len(weights) > 0):
                self.G_ig.es["weight"] = weights

        except Exception as e:
            self.logger.error(f"❌ Error creating iGraph: {e}")
            raise

    # ----------------------------
    # Utility: mapping → membership
    # ----------------------------
    def _mapping_to_membership(self, partition: Dict[Any, Any]) -> List[int]:
        """
        Convert node→label mapping to a membership list aligned with igraph vertex indices.
        Labels can be any hashable (str/int). We convert them to stable ints.
        """
        # Map labels to stable ints
        unique_labels = []
        label_to_int: Dict[Any, int] = {}
        for node_idx in range(len(self.idx_to_node)):
            node = self.idx_to_node[node_idx]
            lbl = partition.get(node, None)
            if lbl not in label_to_int:
                label_to_int[lbl] = len(label_to_int)
                unique_labels.append(lbl)
        membership = [label_to_int[partition.get(self.idx_to_node[i], None)] for i in range(len(self.idx_to_node))]
        return membership

    def modularity(self, partition: Any) -> float:
        """
        Calculate modularity using igraph safely.
        Accepts either:
          - dict(node -> label)
          - list of membership aligned to igraph order
        """
        try:
            if isinstance(partition, dict):
                membership = self._mapping_to_membership(partition)
            else:
                # Assume it is a membership list matching igraph order
                membership = list(partition)
            if self.G_ig is None:
                raise RuntimeError("igraph graph is not initialized.")
            # Use weights if present
            weights = self.G_ig.es["weight"] if "weight" in self.G_ig.es.attributes() else None
            return self.G_ig.modularity(membership, weights=weights)
        except Exception as e:
            self.logger.warning(f"⚠️ Modularity calculation failed: {e}")
            return 0.0

    # ----------------------------
    # Algorithms
    # ----------------------------
    def run_louvain(self) -> Dict[str, Tuple[Dict[Any, int], float]]:
        """Run Louvain algorithm with timing."""
        self.logger.info("🔄 Running Louvain algorithm...")
        results: Dict[str, Tuple[Dict[Any, int], float]] = {}

        try:
            # Validate and standardize edge weights
            self.logger.info("Validating edge weights for Louvain...")
            for u, v, data in self.G_nx.edges(data=True):
                if "weight" in data:
                    if not isinstance(data["weight"], (int, float)) or data["weight"] is None:
                        self.logger.warning(f"Invalid or None weight for edge ({u}, {v}): {data['weight']}. Setting to 1.0.")
                        data["weight"] = 1.0
                else:
                    data["weight"] = 1.0  # Set default weight if missing

            # Try igraph Louvain if available
            if hasattr(self.G_ig, "community_multilevel"):
                self.logger.info("Using igraph Louvain implementation for better performance.")
                # Unweighted
                start_time = time.time()
                part_unweighted = self.G_ig.community_multilevel(weights=None)
                mod_unweighted = self.G_ig.modularity(part_unweighted.membership, weights=None)
                mapping_unweighted = {self.idx_to_node[i]: part_unweighted.membership[i]
                                    for i in range(len(part_unweighted.membership))}
                runtime = time.time() - start_time
                results["louvain_unweighted"] = (mapping_unweighted, mod_unweighted)
                self.logger.info(
                    f"✅ Louvain (unweighted, igraph): {len(set(part_unweighted.membership))} communities, "
                    f"modularity={mod_unweighted:.4f}, time={runtime:.2f}s"
                )
                # Weighted
                start_time = time.time()
                weights = self.G_ig.es["weight"] if "weight" in self.G_ig.es.attributes() else None
                part_weighted = self.G_ig.community_multilevel(weights=weights)
                mod_weighted = self.G_ig.modularity(part_weighted.membership, weights=weights)
                mapping_weighted = {self.idx_to_node[i]: part_weighted.membership[i]
                                    for i in range(len(part_weighted.membership))}
                runtime = time.time() - start_time
                results["louvain_weighted"] = (mapping_weighted, mod_weighted)
                self.logger.info(
                    f"✅ Louvain (weighted, igraph): {len(set(part_weighted.membership))} communities, "
                    f"modularity={mod_weighted:.4f}, time={runtime:.2f}s"
                )
            else:
                self.logger.info("Falling back to python-louvain implementation.")
                # Unweighted
                start_time = time.time()
                part_unweighted = community_louvain.best_partition(self.G_nx, weight=None)
                mod_unweighted = community_louvain.modularity(part_unweighted, self.G_nx, weight=None)
                runtime = time.time() - start_time
                results["louvain_unweighted"] = (part_unweighted, mod_unweighted)
                self.logger.info(
                    f"✅ Louvain (unweighted): {len(set(part_unweighted.values()))} communities, "
                    f"modularity={mod_unweighted:.4f}, time={runtime:.2f}s"
                )
                # Weighted
                start_time = time.time()
                part_weighted = community_louvain.best_partition(self.G_nx, weight="weight")
                mod_weighted = community_louvain.modularity(part_weighted, self.G_nx, weight="weight")
                runtime = time.time() - start_time
                results["louvain_weighted"] = (part_weighted, mod_weighted)
                self.logger.info(
                    f"✅ Louvain (weighted): {len(set(part_weighted.values()))} communities, "
                    f"modularity={mod_weighted:.4f}, time={runtime:.2f}s"
                )

        except Exception as e:
            self.logger.error(f"❌ Louvain algorithm failed: {e}")
            raise

        return results
    def run_leiden(self) -> Dict[str, Tuple[Dict[Any, int], float]]:
        """Run Leiden algorithm with proper error handling."""
        self.logger.info("🔄 Running Leiden algorithm...")
        results: Dict[str, Tuple[Dict[Any, int], float]] = {}

        try:
            # Unweighted
            start_time = time.time()
            part_unweighted = self.G_ig.community_leiden(
                objective_function="modularity",
                weights=None,
                n_iterations=2  # Use n_iterations for determinism instead of seed
            )
            mod_unweighted = part_unweighted.modularity
            mapping_unweighted = {self.idx_to_node[i]: part_unweighted.membership[i]
                                for i in range(len(part_unweighted.membership))}
            runtime = time.time() - start_time
            results["leiden_unweighted"] = (mapping_unweighted, mod_unweighted)
            self.logger.info(
                f"✅ Leiden (unweighted): {len(set(part_unweighted.membership))} communities, "
                f"modularity={mod_unweighted:.4f}, time={runtime:.2f}s"
            )

            # Weighted (if weights exist)
            start_time = time.time()
            weights = self.G_ig.es["weight"] if "weight" in self.G_ig.es.attributes() else None
            part_weighted = self.G_ig.community_leiden(
                objective_function="modularity",
                weights=weights,
                n_iterations=2
            )
            mod_weighted = part_weighted.modularity
            mapping_weighted = {self.idx_to_node[i]: part_weighted.membership[i]
                                for i in range(len(part_weighted.membership))}
            runtime = time.time() - start_time
            results["leiden_weighted"] = (mapping_weighted, mod_weighted)
            self.logger.info(
                f"✅ Leiden (weighted): {len(set(part_weighted.membership))} communities, "
                f"modularity={mod_weighted:.4f}, time={runtime:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"❌ Leiden algorithm failed: {e}")
            raise  # Re-raise to help debugging

        return results

    def run_lpa(self) -> Dict[str, Tuple[Dict[Any, int], float]]:
        """Run Label Propagation Algorithm (unweighted)."""
        self.logger.info("🔄 Running Label Propagation Algorithm (LPA)...")
        results: Dict[str, Tuple[Dict[Any, int], float]] = {}

        try:
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comms = list(nx.algorithms.community.label_propagation_communities(self.G_nx))

            # Convert to mapping
            mapping_unweighted: Dict[Any, int] = {}
            for idx, comm in enumerate(comms):
                for node in comm:
                    mapping_unweighted[node] = idx

            # Compute modularity via igraph (membership aligned)
            mod_unweighted = self.modularity(mapping_unweighted)
            runtime = time.time() - start_time

            results["lpa_unweighted"] = (mapping_unweighted, mod_unweighted)
            self.logger.info(
                f"✅ LPA: {len(comms)} communities, modularity={mod_unweighted:.4f}, time={runtime:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"❌ LPA algorithm failed: {e}")

        return results

    def run_lpa_semi(self, semi_labels_path: str, max_iter: int = 20) -> Dict[str, Tuple[Dict[Any, Any], float]]:
        """
        Run Semi-Supervised Label Propagation Algorithm.
        semi_labels_path: CSV file containing 'peerid' and 'label' columns (from LLM labeling)
        """
        self.logger.info("🔄 Running Semi-Supervised LPA...")

        results: Dict[str, Tuple[Dict[Any, Any], float]] = {}
        try:
            csv_path = Path(semi_labels_path)
            if not csv_path.exists():
                self.logger.warning(f"⚠️ Labels file not found: {semi_labels_path}. Skipping semi-supervised LPA.")
                return results

            df_labels = pd.read_csv(csv_path)
            if "peerid" not in df_labels.columns or "label" not in df_labels.columns:
                self.logger.warning("⚠️ labels CSV must contain 'peerid' and 'label' columns. Skipping.")
                return results

            label_map = dict(zip(df_labels["peerid"], df_labels["label"]))

            # Initialize node labels
            nx.set_node_attributes(self.G_nx, None, "label")
            labeled_set = 0
            for node, lbl in label_map.items():
                if node in self.G_nx:
                    self.G_nx.nodes[node]["label"] = lbl
                    labeled_set += 1
            self.logger.info(f"🔖 Initially labeled nodes: {labeled_set}")

            unlabeled_nodes = [n for n in self.G_nx.nodes if self.G_nx.nodes[n]["label"] is None]

            from collections import Counter
            for it in range(max_iter):
                changes = 0
                for node in unlabeled_nodes:
                    neighbor_labels = [
                        self.G_nx.nodes[nb]["label"]
                        for nb in self.G_nx.neighbors(node)
                        if self.G_nx.nodes[nb]["label"] is not None
                    ]
                    if neighbor_labels:
                        most_common = Counter(neighbor_labels).most_common(1)[0][0]
                        if self.G_nx.nodes[node]["label"] != most_common:
                            self.G_nx.nodes[node]["label"] = most_common
                            changes += 1
                self.logger.info(f"Iteration {it + 1}: {changes} label changes.")
                if changes == 0:
                    break

            # Prepare mapping
            mapping_semi: Dict[Any, Any] = {n: self.G_nx.nodes[n]["label"] for n in self.G_nx.nodes}

            # Encode labels to ints and compute modularity (aligned membership)
            # Reuse modularity(partition dict) for alignment & weights handling
            mod_score = self.modularity(mapping_semi)
            results["lpa_semi"] = (mapping_semi, mod_score)

            self.logger.info(
                f"✅ LPA Semi-Supervised: {len(set(mapping_semi.values()))} communities, modularity={mod_score:.4f}"
            )

        except Exception as e:
            self.logger.error(f"❌ LPA Semi-Supervised algorithm failed: {e}")

        return results

    def run_infomap(self) -> Dict[str, Tuple[Dict[Any, int], float]]:
        """Run Infomap algorithm (weighted if weights exist)."""
        self.logger.info("🔄 Running Infomap algorithm...")
        results: Dict[str, Tuple[Dict[Any, int], float]] = {}

        try:
            start_time = time.time()
            # Use seed for determinism
            im = Infomap("--two-level --silent --seed 42")

            # Add links with indices (use consistent mapping)
            for u, v, data in self.G_nx.edges(data=True):
                ui = self.node_to_idx[u]
                vi = self.node_to_idx[v]
                w = float(data.get("weight", 1.0))
                im.addLink(ui, vi, w)

            im.run()

            # Extract results, map back to original node ids
            mapping_idx_to_module: Dict[int, int] = {node.node_id: node.module_id for node in im.nodes}
            mapping: Dict[Any, int] = {
                self.idx_to_node[i]: mapping_idx_to_module.get(i, -1) for i in range(len(self.idx_to_node))
            }

            # Compute modularity via igraph (aligned)
            mod_score = self.modularity(mapping)
            runtime = time.time() - start_time

            results["infomap_weighted"] = (mapping, mod_score)
            self.logger.info(
                f"✅ Infomap: {len(set(mapping.values()))} communities, modularity={mod_score:.4f}, time={runtime:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"❌ Infomap algorithm failed: {e}")

        return results

    # ----------------------------
    # Persistence
    # ----------------------------
    def save_results(self, algo_results: Dict[str, Tuple[Dict[Any, Any], float]]) -> None:
        """Save results with enhanced formatting."""
        self.logger.info("💾 Saving results...")

        summary_data = []

        for algo_name, (partition_dict, modularity_score) in algo_results.items():
            try:
                # DataFrame of partition
                df_part = pd.DataFrame({
                    "peerid": list(partition_dict.keys()),
                    f"community_{algo_name}": list(partition_dict.values())
                })

                # Merge with group info
                df_merged = self.df_info.merge(df_part, on="peerid", how="inner")

                # Save individual results
                output_path = self.results_dir / f"{algo_name}.csv"
                df_merged.to_csv(output_path, index=False)

                # Collect summary info
                num_communities = len(set(partition_dict.values()))
                summary_data.append({
                    "algorithm": algo_name,
                    "num_communities": num_communities,
                    "modularity": modularity_score,
                    "nodes_processed": len(partition_dict)
                })

                self.logger.info(f"✅ Saved {algo_name}: {num_communities} communities, modularity={modularity_score:.4f}")

            except Exception as e:
                self.logger.error(f"❌ Error saving {algo_name}: {e}")

        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.results_dir / "algorithm_comparison.csv", index=False)
            self.logger.info("📊 Algorithm comparison summary saved")

    # ----------------------------
    # Orchestrator
    # ----------------------------
    def run_all(self, semi_labels_path: Optional[str] = None) -> None:
        """Run all community detection algorithms."""
        total_start_time = time.time()
        self.logger.info("🚀 Starting comprehensive community detection analysis")

        try:
            self.load_data()

            # Run algorithms
            self.all_results.update(self.run_louvain())
            self.all_results.update(self.run_leiden())
            self.all_results.update(self.run_lpa())
            self.all_results.update(self.run_infomap())

            # Semi-supervised LPA only if labels are provided or exist at default path
            if semi_labels_path is None:
                default_labels = self.results_dir / "labels.csv"
                semi_labels_path = str(default_labels) if default_labels.exists() else None

            if semi_labels_path:
                self.all_results.update(self.run_lpa_semi(semi_labels_path=semi_labels_path))

            # Save results
            self.save_results(self.all_results)

            total_runtime = time.time() - total_start_time
            self.logger.info(f"🎉 Analysis completed successfully in {total_runtime:.2f}s")

        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            raise

    def get_best_algorithm(self) -> Tuple[Optional[str], float]:
        """Find the algorithm with highest modularity."""
        if not self.all_results:
            return None, 0.0
        best_algo, (partition, mod) = max(self.all_results.items(), key=lambda x: x[1][1])
        return best_algo, mod


if __name__ == "__main__":
    detector = CommunityDetection(
        graph_path="results/telegram_graph.pkl",
        group_info_path="data/processed/groups_clean.csv"
    )
    # If you have labels at results/labels.csv it will be auto-used; otherwise skipped.
    detector.run_all()

    # Show best algorithm
    best_algo, best_mod = detector.get_best_algorithm()
    print(f"\n🏆 Best algorithm: {best_algo} (Modularity: {best_mod:.4f})")
