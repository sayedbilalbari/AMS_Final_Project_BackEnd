# G-Retriever

[![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)](https://arxiv.org/abs/2402.07630)

This repository contains the source code for the paper ["<u>G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering</u>"](https://arxiv.org/abs/2402.07630).

We introduce **G-Retriever**, a flexible question-answering framework targeting real-world textual graphs, applicable to multiple applications including scene graph understanding, common sense reasoning, and knowledge graph reasoning.
<img src="figs/chat.svg">

**G-Retriever** integrates the strengths of Graph Neural Networks (GNNs), Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG), and can be fine-tuned to enhance graph understanding via soft prompting.
<img src="figs/overview.svg">

## News
[2024.09] [PyG 2.6](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.6.0) now supports **G-Retriever**! 🎉 \[[Dataset](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/web_qsp_dataset.html)\]\[[Model](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GRetriever.html?highlight=gretriever)\]

## Citation
```
@article{he2024g,
  title={G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering},
  author={He, Xiaoxin and Tian, Yijun and Sun, Yifei and Chawla, Nitesh V and Laurent, Thomas and LeCun, Yann and Bresson, Xavier and Hooi, Bryan},
  journal={arXiv preprint arXiv:2402.07630},
  year={2024}
}
```

## Environment setup
```
conda create --name g_retriever python=3.9 -y
conda activate g_retriever

# https://pytorch.org/get-started/locally/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install peft
pip install pandas
pip install ogb
pip install transformers
pip install wandb
pip install sentencepiece
pip install torch_geometric
pip install datasets
pip install pcst_fast
pip install gensim
pip install scipy==1.12
pip install protobuf
```

## Download the Llama 2 Model
1. Go to Hugging Face: https://huggingface.co/meta-llama/Llama-2-7b-hf. You will need to share your contact information with Meta to access this model.
2. Sign up for a Hugging Face account (if you don’t already have one).
3. Generate an access token: https://huggingface.co/docs/hub/en/security-tokens.
4. Add your token to the code file as follows:
  ```
  From transformers import AutoModel
  access_token = "hf_..."
  model = AutoModel.from_pretrained("private/model", token=access_token)
  ```




## Data Preprocessing
```
# expla_graphs
python -m src.dataset.preprocess.expla_graphs
python -m src.dataset.expla_graphs

# scene_graphs, might take
python -m src.dataset.preprocess.scene_graphs
python -m src.dataset.scene_graphs

# webqsp
python -m src.dataset.preprocess.webqsp
python -m src.dataset.webqsp
```

## Training
Replace path to the llm checkpoints in the `src/model/__init__.py`, then run

### 1) Inference-Only LLM
```
python inference.py --dataset scene_graphs --model_name inference_llm --llm_model_name 7b_chat
```
### 2) Frozen LLM + Prompt Tuning
```
# prompt tuning
python train.py --dataset scene_graphs_baseline --model_name pt_llm

# G-Retriever
python train.py --dataset scene_graphs --model_name graph_llm
```

### 3) Tuned LLM
```
# finetune LLM with LoRA
python train.py --dataset scene_graphs_baseline --model_name llm --llm_frozen False

# G-Retriever with LoRA
python train.py --dataset scene_graphs --model_name graph_llm --llm_frozen False
```

## Reproducibility
Use `run.sh` to run the codes and reproduce the published results in the main table.


## KG Inference
```
python inference_large_graph.py     --graph_path dataset/knowledge_graphs/attention_is_all_you_need_kg_expanded.csv     --question "Can you tell me other works of Ashish Vaswani?"     --openai_api_key "" --output_path "outputs/results.json"
```

## VectorDB Inference
```
python inference_large_graph.py     --graph_path "dataset/knowledge_graphs/attention_is_all_you_need_kg_expanded.csv"     --question "What are the main concepts of the attention is all you need paper? Explain in detail."     --openai_api_key ""     --vector_db_path "vectorDB"     --vector_db_collection "RAG-Docs"     --output_path "outputs/results.json"
```

Add pds needed to the dataset/pdfs folder and run the below command to populate the vectorDB.

## VectorDB Load
```
python -m src.utils.load_vector_db     --pdf_dir "dataset/pdfs"     --persist_dir "vectorDB"     --openai_api_key ""     --collection_name "RAG-Docs"     --chunk_size 1000     --chunk_overlap 200
```
