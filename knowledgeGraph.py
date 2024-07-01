from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from pyvis.network import Network
import os
import pickle




def build_graph(documents,llm=None,max_triplets_per_chunk=10,embeddings="microsoft/codebert-base"):
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=max_triplets_per_chunk,
        llm=llm,
        embed_model =HuggingFaceEmbedding(model_name=embeddings),
        storage_context=storage_context,
    )
    print("KG built succesfully!")
    os.makedirs("results", exist_ok=True)
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.show("Graph_visualization.html")
    return index

def save_index(index):
    os.makedirs("results", exist_ok=True)
    with open('results/graphIndex', 'wb') as f:
        pickle.dump(index, f)
    print("Index saved succesfully!")