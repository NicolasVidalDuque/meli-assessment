from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import os
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyvis.network import Network
import matplotlib.pyplot as plt


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class InternalGraphRepresentation:
    """Internal representation of graph edges."""
    edges: List[Tuple[str, str]]
    
    def __len__(self) -> int:
        return len(self.edges)
    
    def is_empty(self) -> bool:
        return len(self.edges) == 0


@dataclass
class GraphMetrics:
    """Container for computed graph metrics."""
    pagerank: Dict[str, float]
    hubs: Dict[str, float]
    authorities: Dict[str, float]
    in_degree: Dict[str, int]
    out_degree: Dict[str, int]


@dataclass
class NodeAnalysis:
    """Analysis output for a node."""
    node: str
    pagerank: float
    authority: float
    hub: float
    in_degree: int
    out_degree: int


# ============================================================================
# EDGE READERS (Strategy Pattern)
# ============================================================================

class EdgeReader(ABC):
    """Abstract base for edge readers."""
    # TODO: Refactor to eliminate inheritance but include composition pattern.
    
    @abstractmethod
    def read(self, file_path: str, max_edges: int) -> InternalGraphRepresentation:
        """Read edges from file."""
        pass
    
    def _parse_line(self, line: str) -> Optional[Tuple[str, str]]:
        """Helper to parse edge lines (skip comments, empty lines)."""
        if line.startswith("#") or not line.strip():
            return None
        parts: List[str] = line.split()
        if len(parts) < 2:
            return None
        return (parts[0], parts[1])


class SimpleEdgeReader(EdgeReader):
    """Sequential read until max_edges limit."""
    
    def read(self, file_path: str, max_edges: int) -> InternalGraphRepresentation:
        edges: List[Tuple[str, str]] = []
        with open(file_path, "r", encoding="utf8", errors="ignore") as f:
            for line in f:
                parsed: Optional[Tuple[str, str]] = self._parse_line(line)
                if parsed:
                    edges.append(parsed)
                    if len(edges) >= max_edges:
                        break
        return InternalGraphRepresentation(edges=edges)


class SnowballEdgeReader(EdgeReader):
    """BFS-based snowball sampling."""
    
    def read(self, file_path: str, max_edges: int) -> InternalGraphRepresentation:
        # First pass: build adjacency
        all_edges: List[Tuple[str, str]] = []
        with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                parsed: Optional[Tuple[str, str]] = self._parse_line(line)
                if parsed:
                    all_edges.append(parsed)
                    if len(all_edges) >= 5_000_000:  # safety limit
                        break
        
        if not all_edges:
            return InternalGraphRepresentation(edges=[])
        
        # Build adjacency dict
        adj_out: Dict[str, List[str]] = {}
        src: str
        dst: str
        for src, dst in all_edges:
            adj_out.setdefault(src, []).append(dst)
        
        # BFS from first node
        start_node: str = all_edges[0][0]
        frontier: Set[str] = {start_node}
        visited: Set[str] = set(frontier)
        sampled_edges: List[Tuple[str, str]] = []
        
        while frontier and len(sampled_edges) < max_edges:
            new_frontier: Set[str] = set()
            node: str
            for node in frontier:
                target: str
                for target in adj_out.get(node, []):
                    sampled_edges.append((node, target))
                    if target not in visited:
                        new_frontier.add(target)
                        visited.add(target)
                    if len(sampled_edges) >= max_edges:
                        break
                if len(sampled_edges) >= max_edges:
                    break
            frontier = new_frontier
        
        return InternalGraphRepresentation(edges=sampled_edges)


# ============================================================================
# GRAPH ABSTRACTION (Adapter Pattern + Strategy Pattern)
# ============================================================================

class GraphOperations(ABC):
    """Abstract interface for graph operations (independent of implementation)."""
    
    @abstractmethod
    def compute_metrics(self) -> GraphMetrics:
        pass
    
    @abstractmethod
    def extract_subgraph_topk_1hop(self, top_n: int) -> Tuple['GraphOperations', pd.DataFrame]:
        """Returns subgraph and ranking of original graph."""
        pass
    
    @abstractmethod
    def get_dataframe_analysis(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def num_nodes(self) -> int:
        pass
    
    @abstractmethod
    def num_edges(self) -> int:
        pass


class NetworkXGraphAdapter(GraphOperations):
    """Encapsulates all NetworkX-dependent logic."""
    
    def __init__(self, graph_repr: InternalGraphRepresentation):
        self._graph = nx.DiGraph()
        self._graph.add_edges_from(graph_repr.edges)
        self._metrics: Optional[GraphMetrics] = None
        self._analysis_df: Optional[pd.DataFrame] = None
    
    def compute_metrics(self) -> GraphMetrics:
        """Compute PageRank, HITS, degrees."""
        if self._metrics is not None:
            return self._metrics
        
        print(f"Computing metrics: {self.num_nodes()} nodes, {self.num_edges()} edges")
        
        # PageRank
        pagerank: Dict[str, float]
        try:
            pagerank = nx.pagerank(self._graph, tol=1e-6, max_iter=200)
        except Exception as e:
            print(f"PageRank fallback: {e}")
            pagerank = nx.pagerank(self._graph, tol=1e-4, max_iter=500)
        
        # HITS
        hubs: Dict[str, float]
        authorities: Dict[str, float]
        try:
            hubs, authorities = nx.hits(self._graph, max_iter=200, tol=1e-8)
        except Exception as e:
            print(f"HITS fallback: {e}")
            hubs = {n: 0.0 for n in self._graph.nodes()}
            authorities = {n: 0.0 for n in self._graph.nodes()}
        
        # Degrees
        in_degree: Dict[str, int] = dict(self._graph.in_degree())
        out_degree: Dict[str, int] = dict(self._graph.out_degree())
        
        self._metrics = GraphMetrics(
            pagerank=pagerank,
            hubs=hubs,
            authorities=authorities,
            in_degree=in_degree,
            out_degree=out_degree
        )
        return self._metrics
    
    def get_dataframe_analysis(self) -> pd.DataFrame:
        """Build ranking dataframe from metrics."""
        if self._analysis_df is not None:
            return self._analysis_df
        
        metrics: GraphMetrics = self.compute_metrics()
        rows: List[NodeAnalysis] = []
        node: str
        for node in self._graph.nodes():
            rows.append(NodeAnalysis(
                node=node,
                pagerank=metrics.pagerank.get(node, 0.0),
                authority=metrics.authorities.get(node, 0.0),
                hub=metrics.hubs.get(node, 0.0),
                in_degree=metrics.in_degree.get(node, 0),
                out_degree=metrics.out_degree.get(node, 0)
            ))
        
        r: NodeAnalysis
        df: pd.DataFrame = pd.DataFrame([
            {
            "node": getattr(r, 'node', "N/A"),
            "pagerank": getattr(r, 'pagerank', 0.0),
            "authority": getattr(r, 'authority', 0.0),
            "hub": getattr(r, 'hub', 0.0),
            "in_degree": getattr(r, 'in_degree', 0),
            "out_degree": getattr(r, 'out_degree', 0)
            }
            for r in rows
        ]).sort_values("pagerank", ascending=False).reset_index(drop=True)
        
        self._analysis_df = df
        return df
    
    def extract_subgraph_topk_1hop(self, top_n: int) -> Tuple['NetworkXGraphAdapter', pd.DataFrame]:
        """Extract Top-K nodes + 1-hop neighborhood."""
        ranking_df: pd.DataFrame = self.get_dataframe_analysis()
        top_nodes: Set[str] = set(ranking_df['node'].iloc[:top_n].tolist())
        
        # Include 1-hop neighbors
        sub_nodes: Set[str] = set(top_nodes)
        node: str
        for node in top_nodes:
            sub_nodes.update(self._graph.predecessors(node))
            sub_nodes.update(self._graph.successors(node))
        
        sub_graph: nx.DiGraph = self._graph.subgraph(sub_nodes).copy()
        sub_repr: InternalGraphRepresentation = InternalGraphRepresentation(edges=list(sub_graph.edges()))
        sub_adapter: NetworkXGraphAdapter = NetworkXGraphAdapter(sub_repr)
        
        return sub_adapter, ranking_df
    
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()
    
    def num_edges(self) -> int:
        return self._graph.number_of_edges()
    
    def get_networkx_graph(self) -> nx.DiGraph:
        """Direct access for visualization (temporary, not ideal)."""
        return self._graph


# ============================================================================
# VISUALIZERS (Strategy Pattern)
# ============================================================================

class GraphVisualizer(ABC):
    """Abstract visualizer."""
    
    @abstractmethod
    def visualize(self, graph_adapter: GraphOperations, output_path: str) -> None:
        pass


class PyvisVisualizer(GraphVisualizer):
    """Interactive HTML visualization."""
    
    def visualize(self, graph_adapter: GraphOperations, output_path: str) -> None:
        ranking_df: pd.DataFrame = graph_adapter.get_dataframe_analysis()
        g: Optional[nx.DiGraph] = graph_adapter.get_networkx_graph() if hasattr(graph_adapter, 'get_networkx_graph') else None
        if g is None or graph_adapter.num_nodes() == 0:
            print("Cannot visualize: empty graph or adapter mismatch.")
            return
        
        net: Network = Network(height="900px", width="100%", directed=True, notebook=False)
        pr_dict: Dict[str, float] = dict(zip(ranking_df['node'], ranking_df['pagerank']))
        indeg_dict: Dict[str, int] = dict(zip(ranking_df['node'], ranking_df['in_degree']))
        
        node: str
        for node in g.nodes():
            pr: float = pr_dict.get(node, 0.0)
            indeg: int = indeg_dict.get(node, 0)
            size: float = max(6, 40 * (pr if pr > 0 else 0.0001))
            title: str = f"node: {node}<br>pagerank: {pr:.6f}<br>in_degree: {indeg}"
            net.add_node(node, label=str(node), title=title, value=pr, size=size)
        
        u: str
        v: str
        for u, v in g.edges():
            net.add_edge(u, v)
        
        net.set_options("""
        var options = {
          "nodes": {"borderWidth":1,"size":25},
          "physics": {"enabled": true}
        }
        """)
        net.save_graph(output_path)
        print(f"Interactive visualization saved to {output_path}")


class MatplotlibVisualizer(GraphVisualizer):
    """Static PNG visualization (small graphs only)."""
    
    def visualize(self, graph_adapter: GraphOperations, output_path: str) -> None:
        n: int = graph_adapter.num_nodes()
        if n == 0:
            print("Skipping static visualization: empty graph.")
            return
        if n > 500:
            print("Too many nodes for static plot. Use interactive visualization.")
            return
        
        g: nx.DiGraph = graph_adapter.get_networkx_graph()
        ranking_df: pd.DataFrame = graph_adapter.get_dataframe_analysis()
        pr_dict: Dict[str, float] = dict(zip(ranking_df['node'], ranking_df['pagerank']))
        
        pos: Dict[str, np.ndarray] = nx.spring_layout(g, k=0.15, iterations=100)
        node: str
        sizes: List[float] = [200 + 3000 * pr_dict.get(node, 0.0) for node in g.nodes()]
        
        plt.figure(figsize=(12, 9))
        nx.draw_networkx_nodes(g, pos, node_size=sizes)
        nx.draw_networkx_edges(g, pos, alpha=0.3)
        nx.draw_networkx_labels(g, pos, font_size=8)
        plt.axis('off')
        plt.title("Subgraph Visualization")
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Static visualization saved to {output_path}")


# ============================================================================
# ORCHESTRATOR (Facade Pattern)
# ============================================================================

class WebGraphAnalyzer:
    """Main orchestrator - ties everything together."""
    
    def __init__(
        self,
        edge_reader: EdgeReader,
        graph_adapter_class=NetworkXGraphAdapter,
        visualizers: Optional[List[GraphVisualizer]] = None
    ):
        self.edge_reader = edge_reader
        self.graph_adapter_class = graph_adapter_class
        self.visualizers = visualizers or [PyvisVisualizer(), MatplotlibVisualizer()]
        self.graph_adapter: Optional[GraphOperations] = None
    
    def analyze(
        self,
        file_path: str,
        max_edges: int = 50000,
        topk: int = 20,
        sub_k: int = 50,
        out_dir: str = "out"
    ) -> pd.DataFrame:
        """Full analysis pipeline."""
        os.makedirs(out_dir, exist_ok=True)
        
        # Read
        print(f"Reading edges from {file_path}...")
        graph_repr: InternalGraphRepresentation = self.edge_reader.read(file_path, max_edges)
        print(f"Loaded {len(graph_repr)} edges")
        
        # Build graph
        self.graph_adapter = self.graph_adapter_class(graph_repr)
        
        # Compute metrics
        ranking_df: pd.DataFrame = self.graph_adapter.get_dataframe_analysis()
        
        # Save Top-K
        topk_df: pd.DataFrame = ranking_df.head(topk)
        csv_path: str = os.path.join(out_dir, f"top_{topk}_pagerank.csv")
        topk_df.to_csv(csv_path, index=False)
        print(f"Saved Top-{topk} to {csv_path}")
        
        # Extract and visualize subgraph
        sub_adapter: GraphOperations
        _: pd.DataFrame
        sub_adapter, _ = self.graph_adapter.extract_subgraph_topk_1hop(top_n=sub_k)
        print(f"Subgraph: {sub_adapter.num_nodes()} nodes, {sub_adapter.num_edges()} edges")
        
        visualizer: GraphVisualizer
        for visualizer in self.visualizers:
            ext: str = "html" if isinstance(visualizer, PyvisVisualizer) else "png"
            out_path: str = os.path.join(out_dir, f"explanatory_subgraph.{ext}")
            visualizer.visualize(sub_adapter, out_path)
        
        print(f"Done. Outputs in {out_dir}")
        return ranking_df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Web graph analysis")
    parser.add_argument("--input", required=True, help="Path to edge list file")
    parser.add_argument("--max-edges", type=int, default=50000, help="Max edges to read")
    parser.add_argument("--sampling", choices=["simple", "snowball"], default="simple")
    parser.add_argument("--topk", type=int, default=20, help="Top-K for CSV")
    parser.add_argument("--sub_k", type=int, default=50, help="Top-K for subgraph")
    parser.add_argument("--out-dir", default="out", help="Output directory")
    args: argparse.Namespace = parser.parse_args()
    
    # Select reader
    reader: EdgeReader = SnowballEdgeReader() if args.sampling == "snowball" else SimpleEdgeReader()
    
    # Create analyzer
    analyzer: WebGraphAnalyzer = WebGraphAnalyzer(edge_reader=reader)
    
    # Run
    analyzer.analyze(
        file_path=args.input,
        max_edges=args.max_edges,
        topk=args.topk,
        sub_k=args.sub_k,
        out_dir=args.out_dir
    )