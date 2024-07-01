from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter
from llama_index.core import Document
from llama_index.core import Settings




def initialize_llm(base_url="http://localhost:11434",model="llama3",chunk_size = 512):
    llm = Ollama(base_url=base_url,model=model)
    Settings.llm = llm
    Settings.chunk_size = chunk_size
    print(f"{model} initialized succesfully!")

def code_spiltting(documents,language="python",):
    splitter = CodeSplitter(
        language=language,
        chunk_lines=30,  # lines per chunk
        chunk_lines_overlap=6,  # lines overlap between chunks
        max_chars=1500,  # max chars per chunk
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"{len(nodes)} nodes created succesfully!")
    return nodes

def convert_nodes_to_docs(nodes):
    documents_from_nodes = [Document(text=node.text, metadata=node.metadata) for node in nodes]
    print(f"{len(documents_from_nodes)} number of documents are being converted successfully!")
    return documents_from_nodes

def load_directory(directory_path,code_file=False,language="python"):

    documents = SimpleDirectoryReader(directory_path).load_data()

    if code_file:
        nodes=code_spiltting(documents,language)
        docs=convert_nodes_to_docs(nodes)
        print(f"{len(documents)}Documents loaded succesfully!")
        return docs

    print(f"{len(documents)}Documents loaded succesfully!")
    return documents


