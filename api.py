from flask import Flask, request, jsonify
from src.dataset.large_graph import LargeGraphDataset
from types import SimpleNamespace
from src.model.text_graph_llm import TextOnlyGraphLLM
from torch.utils.data import DataLoader
from src.utils.collate import collate_fn
from tqdm import tqdm
import warnings
from flask_cors import CORS
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

app = Flask(__name__)
CORS(app)

# Load the model at the start to avoid reloading it for each request
model_args = SimpleNamespace(
    max_txt_len=8000,
    max_new_tokens=256,
    openai_api_key="<open_ai_key>",
    vector_db_path="vectorDB",
    vector_db_collection="RAG-Docs"
)
model = TextOnlyGraphLLM(model_args)
model.eval()


@app.route('/generate-answer', methods=['POST'])
def generate_answer():
    # Parse input parameters
    try:
        data = request.get_json()
        graph_path = "dataset/knowledge_graphs/KG_Graphs.csv"
        question = data.get('question')
        batch_size = data.get('batch_size', 1)

        if not graph_path or not question:
            return jsonify({"error": "graph_path and question are required parameters"}), 400

        # Initialize dataset and dataloader
        dataset = LargeGraphDataset(graph_path, question)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
                    'vector_context': outputs['vector_context'][i]
                })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
