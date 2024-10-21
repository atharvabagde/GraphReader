import os
from collections import defaultdict
import networkx as nx
import yaml
import unicodedata
import re
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from pathlib import Path
from .openai_client import OpenAI_client

class Graph:
    """
    A class to represent a graph constructed from text chunks and their associated atomic facts.

    This class processes text chunks to extract atomic facts, normalizes keys, 
    and builds a graph using the extracted information. It utilizes the OpenAI API 
    for fact extraction and NetworkX for graph representation.

    Attributes:
    ----------
    chunks : dict
        A dictionary of text chunks where the key is the chunk ID.
    api_key : str
        The API key for accessing the OpenAI services.
    k_at_dict : defaultdict
        A dictionary that maps keys to lists of atomic facts and their chunk IDs.
    lem_dict : defaultdict
        A dictionary that maps lemmatized keys to lists of atomic facts.
    clean_dict : defaultdict
        A dictionary that maps cleaned keys to lists of atomic facts.
    graph : networkx.Graph
        The constructed graph representing relationships between keys.

    Methods:
    -------
    _clean_string(text: str) -> str:
        Cleans and normalizes a string by removing accents and special characters.

    _process_k_at(k_at_resp: str, chunk_id: int) -> None:
        Processes atomic facts from the response and updates k_at_dict with chunk IDs.

    _normalize_keys() -> None:
        Normalizes keys using lemmatization and cleaning, then updates the clean_dict.

    _process_chunks() -> None:
        Processes each chunk by extracting atomic facts and normalizing keys.

    _build() -> None:
        Builds the graph by adding nodes and edges based on the cleaned atomic facts.

    export_graph(file_path: str = '', filename: str = 'graph') -> None:
        Exports the constructed graph to a GML file.
    """
    def __init__(self, chunk_dict, openai_api_key):
        self.chunks = chunk_dict
        self.api_key = openai_api_key
        self.k_at_dict = defaultdict(list)
        self.lem_dict = defaultdict(list)
        self.clean_dict = defaultdict(list)
        self.graph = nx.Graph()
        self._load_prompts()
        self._build()
        
    def _load_prompts(self,prompts_file_path = 'graphreader/prompts/prompts.yaml'):
        with open(prompts_file_path, 'r') as file:
            self.prompts = yaml.safe_load(file)
    
    def _clean_string(self,text):
        # Normalize the text to NFD (decomposed form) to separate accents
        text = unicodedata.normalize('NFD', text)
        
        # Remove accents by filtering out characters with a combining mark
        text = ''.join([char for char in text if not unicodedata.combining(char)])
        
        # Use regex to remove apostrophes and other special characters, keeping only alphanumerics and spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        
        # Return the cleaned string
        return text
    def _process_k_at(self,k_at_resp, chunk_id):
        at_dict = defaultdict(list)
        for line in re.split(r'\n+', k_at_resp):
            sub = line.split('|')
            at = sub[0]
            for d_key in sub[1:]:
                d_key = d_key.strip().lower() 
                at_dict[d_key].append(at)
        for a_key in at_dict: 
            self.k_at_dict[a_key].append({'atom_fact':" ".join(at_dict[a_key]),'chunk_id':chunk_id})
    
    def _normalize_keys(self):
        key_list = list(self.k_at_dict.keys())
        lemmatizer = WordNetLemmatizer()
        for d_key in key_list:
            self.lem_dict[lemmatizer.lemmatize(d_key)].extend(self.k_at_dict[d_key])
        
        lem_key_list = list(self.lem_dict.keys())
        for d_key in lem_key_list:
            self.clean_dict[self._clean_string(d_key)].extend(self.lem_dict[d_key])
        
        
        ''' 
        lem_key_list = list(self.lem_dict.keys())
        model = SentenceTransformer('whaleloops/phrase-bert')
        key_embs = model.encode(lem_key_list)
        
        self.key_emb = key_embs
        '''
                
    def _process_chunks(self):
        for key,value in tqdm(self.chunks.items(), desc= 'Processing chunks'):
            gpt_client = OpenAI_client(api_key = self.api_key)
            key_at_facts = gpt_client.get_response(value, sys_prompt=self.prompts['key_atomic_prompt'])
            self.fact = key_at_facts
            self._process_k_at(key_at_facts,chunk_id = key)
        self._normalize_keys()
            
    def _build(self):
        self._process_chunks()
        for key, at_fact in tqdm(self.clean_dict.items(), desc= 'Building graph'):
            self.graph.add_node(key, data=at_fact)
            for other_key in self.clean_dict:
                other_document = self._clean_string(". ".join(
                    [i["atom_fact"] for i in self.clean_dict[other_key]]))
                self_documents = self._clean_string(". ".join(
                    [i["atom_fact"] for i in self.clean_dict[key]]))
                if key != other_key and (key in other_document) and (other_key in self_documents):
                    self.graph.add_edge(key, other_key)
    
    def export_graph(self,file_path='data',filename='graph'):
        file_w_ext = filename + ".gml"

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        full_path = Path(file_path)/file_w_ext
        nx.write_gml(self.graph, full_path)
        print(f"Graph exported to {full_path}")