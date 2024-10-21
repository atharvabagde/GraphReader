
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time
from tqdm import tqdm
from .tools_utils import *
from .Text_encoder import Text_Encoder

class Pinecone_client:
    def __init__(self,**kwargs):
        self.api_key = kwargs.get("api_key")
        self.vector_dimension = kwargs.get('vector_dim',384)
        self.metric = kwargs.get('metric','cosine')
        self.db_cloud = kwargs.get('db_cloud','aws')
        self.cloud_region = kwargs.get('cloud_region','us-east-1')
        self.encoder_model = kwargs.get('encoder_model',SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))
        self.encoder = Text_Encoder(encoder_model=self.encoder_model)
        self._set_client()
        
    def __repr__(self):
        print("Pinecone database client")
        
    def _set_client(self):
        self.client = Pinecone(api_key=self.api_key)
        
    def _create_index(self,index_name):
        self.client.create_index(
            name = index_name,
            dimension = self.vector_dimension,
            metric = self.metric,
            spec = ServerlessSpec(
                cloud = self.db_cloud,
                region = self.cloud_region
            )
        )
        
    def _connect_db(self,index_name):
        self._set_client()
        if index_name not in self.client.list_indexes().names():
            self._create_index(index_name)
            
        time.sleep(2)
        self.index = self.client.Index(index_name)
        time.sleep(1)
        
    def _embed_nodes(self,graph):
        self.embs = self.encoder.get_embeddings([" ".join([x['atom_fact'] for x in node[1]['data']])for node in graph.nodes(data=True)])
        
    def _get_vectors(self,graph):
        final_data = []
        nodes = [node[0] for node in graph.nodes(data=True)]
        self._embed_nodes(graph)
        for i in range(len(graph.nodes())):
            data = {'id': str(i), 'values': self.embs[i], "metadata": {
                "node": nodes[i]}}
            final_data.append(data)
            
        self.vector_list = final_data
        
    def upsert_data(self,index_name,graph,**kwargs):
        
        
        
        self._connect_db(index_name)
        self._get_vectors(graph)
        
        for ids_vectors_chunk in tqdm(chunks(self.vector_list, batch_size=1000),desc='Adding vectors to Pinecone database in batch of 1000'):
            self.index.upsert(vectors=ids_vectors_chunk)
            time.sleep(2)
            
    def query_index(self, query_text,index_name = None ,**kwargs):
        text_emb = self.encoder.get_embeddings(query_text).tolist()
        if index_name:
            self._connect_db(index_name)
        self.query_matches = self.index.query(vector = text_emb, top_k = 20, include_metadata = True, **kwargs)
        return self.query_matches    
        