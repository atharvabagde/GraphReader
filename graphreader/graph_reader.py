import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import json
from .pinecone_client import Pinecone_client
from .openai_client import OpenAI_client
from .tools_utils import *


class GraphReader:
    def __init__(self,
                 graph, 
                 pinecone_api_key,
                 openai_api_key,
                 vect_db_name = 'graph-reader',
                 **kwargs):
        
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key

        self.vect_db = vect_db_name
        self.graph = graph
        self.pine_client = Pinecone_client(api_key = self.pinecone_api_key)
        self.pine_client.upsert_data(self.vect_db,self.graph)
        self.llm = kwargs.get('llm_model',ChatOpenAI(model="gpt-3.5-turbo",api_key=self.openai_api_key))
        self._load_prompts()
        self._load_json_struct()
        
    def _load_prompts(self,prompts_file_path = 'graphreader/prompts/prompts.yaml'):
        with open(prompts_file_path, 'r') as file:
            self.prompts = yaml.safe_load(file)
            
    def _load_json_struct(self,json_file_path = 'sel_nodes_struct.json'):
        with open(json_file_path) as f:
            json_schema = json.load(f)
            
        self.json_schema = json_schema
        
    def _set_rational_plan(self,query):
        gpt_client = OpenAI_client(api_key = self.openai_api_key)
        self.plan = gpt_client.get_response(query, sys_prompt=self.prompts['rational_plan'])

        
    def _shortlist_nodes(self,query):
        self._set_rational_plan(query)
        comp_text = query + " " + self.plan
        matches = self.pine_client.query_index(comp_text,index_name=self.vect_db)
        node_matches = [x['metadata']['node'] for x in matches['matches']]
        self.node_matches = node_matches
        return node_matches
        
    def _format_mssg(self,nodes):
        
        formatted_query = f"""
        Question: {self.query}
        Plan: {self.plan}
        Nodes: {nodes}"""
        
        return formatted_query
        
    
    def _get_initial_nodes(self):
        shortlist_nodes = self._shortlist_nodes(self.query)
        format_query = self._format_mssg(shortlist_nodes)
        llm = self.llm.with_structured_output(self.json_schema)
        chat_template = ChatPromptTemplate.from_messages(
                    [
                        ("system",  self.prompts['select_nodes']),
                        ("user", format_query)
                    ]
                )
        
        messages = chat_template.format_messages()
        resp = llm.invoke(messages)
        
        self.sel_nodes = list(resp.values())
        
    def _select_atomic_facts(self,init_nodes):
        formatted_query = f"""
        Question: {self.query}
        Plan: {self.plan}
        Nodes: {init_nodes}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts['read_at_facts']), 
            ("human", "{input}"), 
            ("placeholder", "{agent_scratchpad}"),
        ])

        tools=[read_node, search_neighbors]
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        sel_chunks = agent_executor.invoke({"input": formatted_query})
        self.sel_at_facts = [int(x) for  x in sel_chunks['output'].strip('[]').split(',')]
        
    def _reading_chunks(self,chunk_ids):
        formatted_query = f"""
        Question: {self.query}
        Plan: {self.plan}
        Nodes: {chunk_ids}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts['read_chunks']), 
            ("human", "{input}"), 
            ("placeholder", "{agent_scratchpad}"),
        ])

        tools=[read_chunk,read_next_chunk, read_prev_chunk,write_notes]
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        final_response = agent_executor.invoke({"input": formatted_query})
        
        self.response = final_response['output']
        
    def get_response(self,query):
        self.query = query
        self._get_initial_nodes()
        self._select_atomic_facts(self.sel_nodes)
        self._reading_chunks(self.sel_at_facts)
        
        return self.response
    
