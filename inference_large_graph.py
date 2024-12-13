from src.dataset.large_graph import LargeGraphDataset
from src.model.text_graph_llm import TextOnlyGraphLLM
from torch.utils.data import DataLoader
from src.utils.collate import collate_fn
import argparse
import json
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--question", type=str, required=True, help="Question text")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--vector_db_path", type=str, required=True, help="Path to vector DB")
    parser.add_argument("--vector_db_collection", type=str, default=None, help="Vector DB collection name")
    parser.add_argument("--output_path", type=str, default="outputs/results.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_txt_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Initialize dataset with question text
    dataset = LargeGraphDataset(args.graph_path, args.question)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = TextOnlyGraphLLM(args)
    model.eval()
    
    # Run inference
    results = []
    for batch in tqdm(dataloader):
        outputs = model.inference(batch)
        for i in range(len(outputs['id'])):
            results.append({
                'id': outputs['id'][i],
                'question': outputs['question'][i],
                'prediction': outputs['pred'][i],
                'subgraph_desc': outputs['desc'][i],
                # 'original_graph': outputs['original_graph'][i],
                'vector_context': outputs['vector_context'][i]
            })
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 