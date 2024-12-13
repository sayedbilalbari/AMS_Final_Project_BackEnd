import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import pandas as pd


def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    # Get the device from the input graph
    device = graph.x.device
    
    # Move q_emb to the same device as graph
    q_emb = q_emb.to(device)
    
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)
        
        n_prizes = torch.zeros_like(n_prizes)
        prizes_range = torch.arange(topk, 0, -1, device=device).float()
        n_prizes[topk_n_indices] = prizes_range
    else:
        n_prizes = torch.zeros(graph.num_nodes, device=device)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))
        
        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges, device=device)

    # Move tensors to CPU before numpy operations
    n_prizes = n_prizes.cpu()
    e_prizes = e_prizes.cpu()
    edge_index = graph.edge_index.cpu()

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    
    for i, (src, dst) in enumerate(edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes.numpy(), np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].cpu().tolist()]
    dst = [mapping[i] for i in edge_index[1].cpu().tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree
from pcst_fast import pcst_fast

def add_connectivity_score(graph, n_prizes, edge_index):
    """Add degree-based importance to node prizes."""
    deg = degree(edge_index[0], graph.num_nodes)
    deg_norm = deg / deg.max()
    connectivity_weight = 0.2  # Adjustable parameter
    return n_prizes + connectivity_weight * deg_norm

def adaptive_edge_cost(e_prizes, percentile=75):
    """Compute adaptive edge cost based on prize distribution."""
    return torch.quantile(e_prizes[e_prizes > 0], percentile/100)

def get_k_hop_neighbors(edge_index, node_indices, k=1):
    """Get k-hop neighbors of given nodes."""
    neighbors = set(node_indices)
    frontier = set(node_indices)

    for _ in range(k):
        new_frontier = set()
        for edge in edge_index.T:
            src, dst = edge.tolist()
            if src in frontier:
                new_frontier.add(dst)
            if dst in frontier:
                new_frontier.add(src)
        frontier = new_frontier - neighbors
        neighbors.update(frontier)

    return list(neighbors)

def compute_neighbor_scores(graph, neighbor_indices, q_emb):
    """Compute relevance scores for neighboring nodes."""
    neighbor_embeddings = graph.x[neighbor_indices]
    return torch.nn.CosineSimilarity(dim=-1)(q_emb.unsqueeze(0), neighbor_embeddings)

def filter_subgraph(data, q_emb, min_size=3, max_size=20):
    """Ensure subgraph meets size constraints while maintaining relevance."""
    if data.num_nodes < min_size:
        # Add highest scoring neighboring nodes
        current_nodes = set(range(data.num_nodes))
        neighbors = get_k_hop_neighbors(data.edge_index, list(current_nodes), k=1)
        new_neighbors = [n for n in neighbors if n not in current_nodes]

        if new_neighbors:
            scores = compute_neighbor_scores(data, new_neighbors, q_emb)
            _, top_indices = torch.topk(scores, min(len(new_neighbors), min_size - data.num_nodes))
            top_neighbors = [new_neighbors[i] for i in top_indices.tolist()]

            # Update graph with new nodes
            new_nodes = list(current_nodes) + top_neighbors
            new_edges = []
            for edge in data.edge_index.T:
                src, dst = edge.tolist()
                if src in new_nodes and dst in new_nodes:
                    new_edges.append([src, dst])

            if new_edges:
                data.edge_index = torch.tensor(new_edges).T
                data.x = data.x[new_nodes]
                data.num_nodes = len(new_nodes)

    elif data.num_nodes > max_size:
        # Compute node importance scores
        node_scores = torch.nn.CosineSimilarity(dim=-1)(q_emb.unsqueeze(0), data.x)

        # Keep top nodes while ensuring connectivity
        _, top_indices = torch.topk(node_scores, max_size)
        top_nodes = set(top_indices.tolist())

        # Ensure connectivity by adding necessary bridge nodes
        components = find_connected_components(data.edge_index, data.num_nodes)
        for component in components:
            if any(node in top_nodes for node in component):
                top_nodes.update(component[:min(len(component), max_size - len(top_nodes))])

        # Update graph with filtered nodes
        new_nodes = list(top_nodes)
        node_map = {old: new for new, old in enumerate(new_nodes)}
        new_edges = []
        for edge in data.edge_index.T:
            src, dst = edge.tolist()
            if src in top_nodes and dst in top_nodes:
                new_edges.append([node_map[src], node_map[dst]])

        if new_edges:
            data.edge_index = torch.tensor(new_edges).T
            data.x = data.x[new_nodes]
            data.num_nodes = len(new_nodes)

    return data

def find_connected_components(edge_index, num_nodes):
    """Find connected components in the graph."""
    components = []
    visited = set()

    def dfs(node, component):
        component.append(node)
        visited.add(node)
        for edge in edge_index.T:
            src, dst = edge.tolist()
            if src == node and dst not in visited:
                dfs(dst, component)
            elif dst == node and src not in visited:
                dfs(src, component)

    for node in range(num_nodes):
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)

    return components

def improved_retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges,
                              topk=12, topk_e=15, temperature=0.1,
                              connectivity_weight=0.2, cost_e=0.5):
    """
    Improved version of retrieval via Prize-Collecting Steiner Tree.
    Args:
        graph: Input graph
        q_emb: Query embedding
        textual_nodes: DataFrame containing node text information
        textual_edges: DataFrame containing edge text information
        topk: Number of top nodes to consider
        topk_e: Number of top edges to consider
        temperature: Temperature parameter for softmax
        connectivity_weight: Weight for connectivity score
        cost_e: Base edge cost
    """
    # Get the device from the input graph
    device = graph.x.device

    # Move q_emb to the same device as graph
    q_emb = q_emb.to(device)

    # Handle empty graph case
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    # Compute node prizes with connectivity consideration
    n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
    n_prizes = add_connectivity_score(graph, n_prizes, graph.edge_index)

    if topk > 0:
        topk = min(topk, graph.num_nodes)
        values, indices = torch.topk(n_prizes, topk)
        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[indices] = torch.softmax(values / temperature, dim=0) * topk
    else:
        n_prizes = torch.zeros(graph.num_nodes, device=device)

    # Compute edge prizes with adaptive costs
    e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
    if topk_e > 0:
        adaptive_cost = adaptive_edge_cost(e_prizes)
        cost_e = min(cost_e, adaptive_cost.item())

        topk_e = min(topk_e, e_prizes.unique().size(0))
        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)

        # Smooth prize assignment
        e_prizes_new = torch.zeros_like(e_prizes)
        for k, threshold in enumerate(topk_e_values):
            mask = e_prizes >= threshold
            e_prizes_new[mask] = torch.softmax(e_prizes[mask] / temperature, dim=0) * (topk_e - k)
        e_prizes = e_prizes_new
    else:
        e_prizes = torch.zeros(graph.num_edges, device=device)

    # Move tensors to CPU before numpy operations
    n_prizes = n_prizes.cpu()
    e_prizes = e_prizes.cpu()
    edge_index = graph.edge_index.cpu()

    # Prepare PCST input
    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}

    # Process edges with local importance consideration
    for i, (src, dst) in enumerate(edge_index.T.numpy()):
        prize_e = e_prizes[i]
        local_importance = (n_prizes[src] + n_prizes[dst]) / 2
        adjusted_cost = cost_e * (1 - local_importance * 0.3)  # 0.3 is importance factor

        if prize_e <= adjusted_cost:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(adjusted_cost - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - adjusted_cost)

    # Prepare final PCST inputs
    prizes = np.concatenate([n_prizes.numpy(), np.array(virtual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs)
        edges = np.array(edges + virtual_edges)

    # Run PCST
    vertices, edges = pcst_fast(
        edges,
        prizes,
        costs,
        -1,  # unrooted
        1,   # num_clusters
        'gw', # pruning
        0    # verbosity_level
    )

    # Process PCST output
    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    # Create output subgraph
    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([
        selected_nodes,
        edge_index[0].cpu().numpy(),
        edge_index[1].cpu().numpy()
    ]))

    # Prepare textual description
    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    # Create final subgraph
    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}
    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].cpu().tolist()]
    dst = [mapping[i] for i in edge_index[1].cpu().tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    # Apply final filtering
    data = filter_subgraph(data, q_emb)

    return data, desc



