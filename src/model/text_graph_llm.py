from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from typing import Dict, List
import pandas as pd
import openai
from src.utils.vector_db import VectorDBRetriever 


class TextOnlyGraphLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Set OpenAI API key
        openai.api_key = args.openai_api_key
        
        # Initialize vector DB retriever
        self.vector_retriever = VectorDBRetriever(
            persist_directory=args.vector_db_path,
            openai_api_key=args.openai_api_key,
            collection_name=args.vector_db_collection
        )
        
        # Config
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.model = "gpt-4-turbo"
        
    def create_prompt(self, graph_text: str, vector_context: List[str], question: str) -> str:
        """Creates a formatted prompt combining graph info, vector DB context and question"""
        vector_context_text = "\nAdditional Context from Knowledge Base:\n" + "\n".join(vector_context)
        
        return f"""Below is a portion of a knowledge graph relevant to the question.
The graph is represented as a set of nodes and their relationships.

{graph_text}

{vector_context_text}

Given this information, please answer the following question:
{question}

Please base your answer on both the graph information and additional context provided.
Answer: """

    @torch.no_grad()
    def inference(self, samples: Dict) -> Dict:
        batch_size = len(samples['id'])
        predictions = []
        vector_contexts = []
        
        for i in range(batch_size):
            # Get vector DB context
            vector_context = self.vector_retriever.retrieve(samples['question'][i], k=3)
            vector_contexts.append(vector_context)
            
            prompt = self.create_prompt(
                samples['desc'][i], 
                vector_context,
                samples['question'][i]
            )
            
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided graph information and additional context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_new_tokens,
                    temperature=0.7
                )
                prediction = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error with GPT-4 API call: {e}")
                prediction = "Error: Could not generate response"
                
            predictions.append(prediction)

        return {
            'id': samples['id'],
            'pred': predictions,
            'question': samples['question'],
            'desc': samples['desc'],
            'original_graph': samples['original_graph'],
            'vector_context': vector_contexts
        }

    def format_graph_text(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> str:
        """
        Format graph components into readable text
        Args:
            nodes_df: DataFrame containing node information
            edges_df: DataFrame containing edge information
        """
        text = "Nodes:\n"
        for _, node in nodes_df.iterrows():
            text += f"- {node['node_text']}\n"
        
        text += "\nRelationships:\n"
        for _, edge in edges_df.iterrows():
            text += f"- {edge['src']} {edge['edge_attr']} {edge['dst']}\n"
            
        return text

    def save_pretrained(self, path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, args):
        """Load model from path"""
        instance = cls(args)
        instance.model = AutoModelForCausalLM.from_pretrained(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        return instance 