import itertools
from langchain_core.tools import tool
import networkx as nx
import pickle


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))
        
@tool
def read_node(node_name:str):
    """
    Read the node from graph and extract atomic facts.

    Args:
        node_name (str): The name of the node whose associated chunk IDs are to be retrieved.

    Returns:
        List[]: A list of dictionaries containing atomic facts and its respective chunk id.
    """
    print(f"Reading atomic facts for node: {node_name}") 
    graph = nx.read_gml('graph.gml')
    return graph.nodes[node_name]['data']

@tool
def search_neighbors(node_name:str):
    """
    Searches for neighboring nodes of the given input node .

    Args:
        node_name (str): The name of the node whose neighbors are to be retrieved.

    Returns:
        List[]: A list of nodes where each node has a list of dictionaries with atomic facts and chunk id.
    """
    print(f"Searching neighbors for node: {node_name}") 
    graph = nx.read_gml('graph.gml')
    neighbor_list =  graph.neighbors[node_name]['data']
    return [graph.nodes[x]['data'] for x in neighbor_list]


@tool
def read_chunk(chunk_id:int):
    """
    Retrieves the original text chunk from the given chunk id.

    Args:
        chunk_id (int): Chunk id of the original text chunk.

    Returns:
        str: text chunk corresponding to the chunk id.
    """
    print(f"Retrieving chunk id: {chunk_id}") 
    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    return chunks[int(chunk_id)]


@tool
def read_next_chunk(chunk_id:int):
    """
    Retrieves the next text chunk from the given chunk id.

    Args:
        chunk_id (int): Reference chunk id.

    Returns:
        str: text chunk corresponding to the next chunk id.
    """
    print(f"Retrieving chunk id: {int(chunk_id)+1}") 
    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    return chunks[int(chunk_id)+1]

@tool
def read_prev_chunk(chunk_id:int):
    """
    Retrieves the previous text chunk from the given chunk id.

    Args:
        chunk_id (int): Reference chunk id.

    Returns:
        str: text chunk corresponding to the previous chunk id.
    """
    print(f"Retrieving chunk id: {int(chunk_id)-11}") 
    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    return chunks[int(chunk_id)-1]

@tool
def write_notes(text: str):
    """
    Appends the insights drawn from chunks by the model to a text file.

    Args:
        text (str): Insights to be appended to the file.
    """
    with open("text.txt", "a") as myfile:
        myfile.write(text)


