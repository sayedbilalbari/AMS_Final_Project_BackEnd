import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.dataset.utils.retrieval import improved_retrieval_via_pcst
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Data

class LargeGraphDataset(Dataset):
    def __init__(self, graph_path, question_text, model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.graph_path = graph_path
        self.question_text = question_text
        
        # Load embedding model for PCST (on CPU)
        self.device = torch.device("cpu")
        self.embed_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load and process the large graph
        self.load_graph()
        
    def load_graph(self):
        """Load and process the large graph"""
        # Read the CSV file
        df = pd.read_csv(self.graph_path)
        
        # Create unique nodes mapping
        unique_nodes = pd.concat([
            df['src entity'], 
            df['destination']
        ]).unique()
        self.node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
        
        # Create textual representations
        self.textual_nodes = pd.DataFrame({
            'node_id': range(len(unique_nodes)),
            'node_attr': unique_nodes
        })
        
        self.textual_edges = pd.DataFrame({
            'src': [self.node_to_idx[src] for src in df['src entity']],
            'edge_attr': df['relationship'],
            'dst': [self.node_to_idx[dst] for dst in df['destination']]
        })
        
        # Create embeddings
        print("Creating node embeddings...")
        node_embeddings = self.text2embedding(self.textual_nodes['node_attr'].tolist())
        
        print("Creating edge embeddings...")
        edge_embeddings = self.text2embedding(self.textual_edges['edge_attr'].tolist())
        
        # Create edge index tensor
        edge_index = torch.tensor([
            self.textual_edges['src'].tolist(),
            self.textual_edges['dst'].tolist()
        ], dtype=torch.long)
        
        # Create PyG graph (on CPU)
        self.graph = Data(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=edge_embeddings,
            num_nodes=len(unique_nodes)
        )
    
    def text2embedding(self, texts, batch_size=32):
        """Convert texts to embeddings using batched processing"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.embed_tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def get_question_embedding(self, question):
        """Get embedding for a question"""
        inputs = self.embed_tokenizer(
            question,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

    def __len__(self):
        return 1

    def format_graph_text(self, df):
        """Format the graph data as readable text"""
        text = "Graph Structure:\n"
        text += "\nNodes:\n"
        for _, row in df.iterrows():
            text += f"- {row['src entity']} -> {row['relationship']} -> {row['destination']}\n"
        return text

    def __getitem__(self, index):
        question = f'Question: {self.question_text}\n\nAnswer:'
        
        # Get question embedding
        q_emb = self.get_question_embedding(self.question_text)
        
        # Get original graph text for debugging
        original_graph_text = self.format_graph_text(pd.read_csv(self.graph_path))
        
        # Get subgraph using PCST with larger topk for better coverage
        subgraph, desc = improved_retrieval_via_pcst(
            graph=self.graph,
            q_emb=q_emb,
            textual_nodes=self.textual_nodes,
            textual_edges=self.textual_edges,
            # topk=30,  # Increased from 15
            # topk_e=20,  # Increased from 10
            # cost_e=0.2,  # Further reduced to allow more edge
        )
        
        return {
            'id': index,
            'question': question,
            'graph': subgraph,
            'desc': desc,
            'original_graph': original_graph_text
        }