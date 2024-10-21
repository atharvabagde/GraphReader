from pypdf import PdfReader
from tqdm import tqdm
import re
import os
import pickle
from pathlib import Path

class Document:
    """
    A class to represent a document and handle text extraction, processing, and chunking.

    Attributes:
    ----------
    path : str
        The file path of the PDF document.
    chunk_len : int
        The maximum length of each text chunk.
    document : PdfReader
        The PdfReader object used to read the PDF document.
    text : str
        The extracted text from the PDF document.
    chunks : dict
        A dictionary of text chunks, indexed by their order.

    Methods:
    -------
    __init__(doc_path: str, chunk_len: int = 1000)
        Initializes the Document object with a file path and chunk length.
        
    __repr__()
        Returns a string representation of the Document object.
    
    _del_head_foot(text, st_ind=0, end_ind=0)
        Removes headers and footers from the text based on provided start and end indices.
        
    _get_text(**kwargs)
        Extracts text from the PDF document and processes it according to the given parameters.
        
    get_chunks(**kwargs)
        Splits the processed text into chunks of specified length and stores them in a dictionary.
        
    export_chunks(export_path='', filename='chunks')
        Exports the chunks as a pickle file to the specified path and filename.
    """
    def __init__(self, doc_path: str, chunk_len: int = 1000):
        # Initialize the document with path and chunk length
        self.path = doc_path
        self.chunk_len = chunk_len
        try:
            self.document = PdfReader(doc_path)
        except Exception as e:
            raise ValueError(f"Error reading the PDF file at {doc_path}: {e}")

    def __repr__(self):
        return "A Document object"

    def _del_head_foot(self, text, st_ind=0, end_ind=0):
        # Remove headers and footers based on start and end indices
        if end_ind == 0:
            end_ind = len(text)
        text = text[st_ind:end_ind]
        return text

    def _get_text(self, **kwargs):
        # Extract and process text from all pages of the document
        text = ''
        for page in tqdm(self.document.pages, desc='Extracting text from document'):
            pg_text = page.extract_text()
            if pg_text:
                proc_text = self._del_head_foot(pg_text, **kwargs)
                text += ' ' + proc_text
            else:
                print("Warning: A page in the document returned no text.")
        self.text = text

    def get_chunks(self, **kwargs):
        # Split the text into chunks of specified length
        self._get_text(**kwargs)
        sentences = re.split(r'(?<=[.!?]) +', self.text)
        chunks = []
        current_chunk = []

        for sentence in tqdm(sentences, desc='Chunking the text'):
            # Add sentences to the current chunk until the length limit is reached
            if len(' '.join(current_chunk + [sentence])) <= self.chunk_len:
                current_chunk.append(sentence)
            else:
                # Save the current chunk and start a new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]

        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Store and return chunks as a dictionary
        self.chunks = {i: chunk for i, chunk in enumerate(chunks) if chunk}
        return self.chunks

    def export_chunks(self, export_path='data', filename='chunks'):
        # Export chunks to a pickle file
        if not hasattr(self, 'chunks'):
            print("No chunks to export. Please generate chunks first using get_chunks() method.")
            return

        file_w_ext = filename + ".pkl"
        if not os.path.exists(export_path):
            os.makedirs(export_path)
            
        full_path = Path(export_path) / file_w_ext
        try:
            with open(full_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            print(f"Chunks exported successfully to {full_path}")
        except Exception as e:
            print(f"Error exporting chunks: {e}")

