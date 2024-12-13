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
