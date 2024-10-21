# GraphReader

## Overview

GraphReader is a graph-based agent system designed to enhance long-context capabilities for Large Language Models (LLMs). This project is based on the paper "GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models." The system aims to address the challenges of processing long and complex texts by structuring the content into a graph and using an LLM agent to explore the graph autonomously.

GraphReader employs Retrieval-Augmented Generation (RAG) techniques to break down large texts into manageable chunks, store them in a graph, and retrieve relevant information based on user queries. It leverages advanced AI tools like Langchain, Pinecone for vector searches, and OpenAI's GPT-3.5-turbo, allowing for efficient query processing over long input contexts.

## Key Features

- **Handles Long Contexts:** Processes long documents by structuring them into a graph for efficient exploration and retrieval.
- **RAG Techniques:** Uses Retrieval-Augmented Generation (RAG) techniques to fetch relevant graph nodes based on user queries.
- **LLM Integration:** Employs GPT-3.5-turbo for reasoning, planning, and answering user queries by exploring the graph.
- **Graph-based Exploration:** Structures large texts as a graph, enabling coarse-to-fine exploration of nodes and their relationships.
- **Predefined Toolset:** Utilizes a set of predefined functions to read nodes and their neighbors, optimizing the exploration process.
  
### Installation

1. Install graphreader:

```bash
pip install graphreader
```

### Example Usage

1. Initialize the document and graph:

```python
from graphreader.document import Document
from graphreader.graph_class import Graph
from graphreader.graph_reader import GraphReader

file_path = 'path/to/your/document.doc.pdf'
doc = Document(file_path)
chunk_dict = doc.get_chunks(st_ind=705)
g = Graph(chunk_dict, openai_api_key = "your_api_key")
doc.export_chunks()
g.export_graph()
```

2. Initialize GraphReader and query the system:

```python
g_reader = GraphReader(graph=g, vect_db_name='graph-reader-test5', pinecone_api_key = "pinecone_api_key", openai_api_key="openai_api_key")

user_query = input(prompt='Enter your query: ')
response = g_reader.get_response(query=user_query)
print(response)
```

## How It Works

### GraphReader Workflow

1. **Document Chunking:**
   The text document is chunked into smaller pieces for easy processing. This is done using the `Document` class, which extracts content from the file and chunks it.

2. **Graph Creation:**
   The chunks are organized into a graph structure, where each node represents a section of the document. The graph is then stored in a Pinecone vector database for efficient querying.

3. **LLM-Based Querying:**
   When a user provides a query, the LLM agent reasons through the content in the graph. It selects relevant nodes, reads their content, and navigates through the graph to generate a coherent response.

4. **Response Generation:**
   The agent explores the graph in a step-by-step manner using a rational plan. It reads node contents, neighbors, and continues gathering information until it has enough data to answer the query.

## Experimental Results

Based on the paper, GraphReader demonstrates superior long-context processing capabilities. The agent consistently outperforms models like GPT-4-128k across a variety of benchmarks. For context lengths ranging from 16k to 256k, GraphReader shows a large margin of improvement, highlighting its ability to handle complex and extended inputs efficiently.

## Tools and Techniques

- **Langchain:** Provides the framework for LLM interaction and tool usage.
- **Pinecone:** Vector database used to store and query the graph data.
- **OpenAI GPT-3.5-turbo:** LLM used for reasoning, planning, and generating responses.
- **RAG (Retrieval-Augmented Generation):** The technique used to retrieve relevant nodes from the graph based on the user’s query.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---
