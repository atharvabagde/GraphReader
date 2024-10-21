from sentence_transformers import SentenceTransformer

class Text_Encoder:
    def __init__(self, encoder_model: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')):
        self.encoder = encoder_model
        
    def _encode_text(self, text: str):
        self.embeddings = self.encoder.encode(text)
        
    def get_embeddings(self, text: str):
        self._encode_text(text)
        return self.embeddings            