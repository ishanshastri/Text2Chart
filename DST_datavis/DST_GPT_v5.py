"""traced version"""
"""endpoint implementation"""
from typing import List, Optional, Any

# import dspy
import sys
import os

from DST_rag_utils import *#format_returned_guidelines_list
import pandas as pd
import re
import os
import numpy as np
from langchain_chroma import Chroma
import json
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from typing import List, Union, Optional
from chart_execution import *
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from dotenv import load_dotenv
load_dotenv()
# Setup support for tracing
from langchain.callbacks import tracing_v2_enabled
from langsmith import Client
LANGCHAIN_KEY = os.getenv('LANGCHAIN_KEY')
print(LANGCHAIN_KEY)

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_KEY
client = Client()

"""defaults"""
HUGFACE_TOKEN = os.getenv('HUGFACE_TOKEN')
GOOGLE_TOKEN = os.getenv('GOOGLE_TOKEN')
OPENAI_TOKEN = os.getenv('OPENAI_TOKEN')

DEBUG = os.getenv('DEBUG')

DEFAULT_LLM_CONFIGS = {
        'mistral_7b':{
            'repo_id':"mistralai/Mistral-7B-Instruct-v0.2",
            'max_new_tokens':400,
            'min_new_tokens':2,
            'temperature':0.001,
            'huggingfacehub_api_token':HUGFACE_TOKEN,
            'seed':1
        },
        'gemini_pro':{
            "model":"gemini-pro",
            "google_api_key":GOOGLE_TOKEN,
            "temperature":0,
            'seed':1
        },
        'gemini-pro':{
            "model":"gemini-pro",
            "google_api_key":GOOGLE_TOKEN,
            "temperature":0,
            'seed':1
        },
        'huggingface':{
            'repo_id':"mistralai/Mistral-7B-Instruct-v0.2",
            'max_new_tokens':600,
            'min_new_tokens':2,
            'temperature':0.001,
            'huggingfacehub_api_token':HUGFACE_TOKEN,
            'seed':1
        },
        'google':{
            "model":"gemini-pro",
            "google_api_key":GOOGLE_TOKEN,
            "temperature":0,
            'seed':1
        },
        'gpt-4o-mini-2024-07-18':{
            "model":"gpt-4o-mini-2024-07-18",
            "openai_api_key":OPENAI_TOKEN,
            "temperature":0,
            'seed':1,
            'max_tokens':500
        },
        'gpt-4o-mini':{
            "model":"gpt-4o-mini",
            "openai_api_key":OPENAI_TOKEN,
            "temperature":0,
            'seed':1,
            'max_tokens':500
        },
        'openai':{
            "model":"gpt-4o-mini",
            "openai_api_key":OPENAI_TOKEN,
            "temperature":0,
            'seed':1,
            'max_tokens':500
        },
    }

LLM_ENDPOINTS = {
    'mistral_7b':HuggingFaceEndpoint,
    'huggingface':HuggingFaceEndpoint,
    'gemini_pro':GoogleGenerativeAI,
    'gemini-pro':GoogleGenerativeAI,
    'google':GoogleGenerativeAI,
    'gpt':ChatOpenAI,
    'openai':ChatOpenAI
}

class StoryTeller:
    def __init__(self, chain = None, 
                 df:pd.DataFrame=None, 
                 llm_name:Union[str|None]="mistral_7b",
                 llm=None,
                 llm_config:Union[dict|None]=None,
                 debug_llm=None,
                 use_default_llm=False,
                 retriever=None,
                 db_file_path=None,
                 accepted_chart_types=['bar', 'scatter', 'line', 'candlestick', 'heatmap'],
                 steps_prompt=None, code_prompt=None, chart_selector_prompt=None
                 ):
        
        if llm:
            self.llm = llm
        elif llm_config:
            assert llm_name, "if using llm_config, accompanying name must be provided"
            self.llm = LLM_ENDPOINTS[llm_name](**llm_config)
        elif llm_name:
            self.llm = LLM_ENDPOINTS[llm_name](**DEFAULT_LLM_CONFIGS[llm_name])

        self.df = df

        self.history = list()
        self.prompts = list()

        self.get_steps_prompt = steps_prompt
        self.get_code_prompt = code_prompt
        self.get_chart_selector_prompt = chart_selector_prompt

        self.accepted_chart_types = accepted_chart_types
        self.debug_llm = self.llm if debug_llm is None else debug_llm

        assert retriever or db_file_path, "one of retriever or filepath to load retriever from should be sent."
        retriever = retriever if retriever else get_langchain_retriever_from_json_file(db_file_path)
        self.retriever = retriever

        assert (self.llm and retriever) or chain, "either chain or remaining chain configuration must be sent to StoryTeller at init"
        self.chain = self.init_chain(llm=llm, retriever=retriever) if chain is None else chain
        self.debug_chain = self.init_debug_chain(llm=self.debug_llm)

    def code_generate(self, prompt:Optional[str|dict], llm=None, library='plotly', get_trace_url=False):
        chain = self.init_chain(llm) if llm else self.chain
        if not chain:
            print("no chain initialized; you can rectify this by passing an llm")
            return None

        if self.df is None:
            msg="No dataframe set. Please set dataframe df for StoryTeller"
            print(msg)
            raise Exception(msg)
            return None
        
        if self.llm is None:
            print("No llm set; use `set_llm` to set the llm.")
            return None
        
        str_cols = ", ".join([str(k)+ ": " + str(v) for k, v in dict(self.df.dtypes).items()])
        if isinstance(prompt, str):
            prompt = {'question':prompt,
                        'columns_and_types':str_cols,
                        'num_cols':len(self.df.columns),
                        'library':library,
                        'chart_types':str(self.accepted_chart_types)}
        with tracing_v2_enabled() as cb:
            response = chain.invoke(prompt)
            url = cb.get_run_url()
        response_processed = self._clean_response(response=response)

        return (response_processed, url) if get_trace_url else response_processed
    
    def generate_and_execute(self, prompt:Optional[str|dict], llm=None, library='plotly', get_trace_url=False, get_traceback=True):
        code, url = self.code_generate(prompt, llm=llm, library=library, get_trace_url=True)

        output = {}
        output['code'] = code
        if get_trace_url:
            output['url'] = url

        output = execute_code_dict(code=code, vars={'df':self.df.copy(deep=True), 'pd':pd}, get_traceback=get_traceback)
        return output
    
    def generate_with_lib_fallbacks(self, prompt:str, libs:list, get_trace_url=False, get_traceback=True):
        for lib in libs:
            if get_trace_url:
                resp, url = self.code_generate(prompt=prompt, library=lib, get_trace_url=get_trace_url)
                print(url)
            else:
                resp = self.code_generate(prompt=prompt, library=lib, get_trace_url=get_trace_url)
            output = execute_code_dict(code=resp, vars={'df':self.df.copy(deep=True), 'pd':pd}, get_traceback=get_traceback) #, 'np':np
            if not output['err']:
                break
            
            print(f"Error: {str(output['err'])} while using {lib}; might retry.")

        if get_trace_url:
            output['url'] = url

        return output
    
    def generate_with_fallback(self, prompt:str, llm=None, max_attempts=3, library='plotly', get_trace_url=False):
        err = False
        llm = llm if llm else self.llm
        if get_trace_url:
                resp, url = self.code_generate(prompt=prompt, library=library, get_trace_url=get_trace_url)
                print(url)
        else:
            resp = self.code_generate(prompt=prompt, library=library, get_trace_url=get_trace_url)
        
        output = execute_code_dict(code=resp, vars={'df':self.df.copy(deep=True), 'pd':pd}, get_traceback=True)
        i=0
        while output['err'] and i < max_attempts:
            resp = self.debug_chain.invoke({
                'prev_code':resp,
                'traceback':output['err']
            })
            print("debug response: ", resp)
            output = execute_code_dict(code=self._clean_response(response=resp), vars={'df':self.df.copy(deep=True), 'pd':pd}, get_traceback=True)
            i+=1

        if get_trace_url:
            output['url'] = url

        return output
    
    def generate_with_llm_fallback(self, prompt:str, library='plotly', llms_and_configs:dict = None, get_trace_url=False, get_traceback=True):#llms:list=['mistral_7b', 'gemini-pro'], llm_configs:dict=None):
        llms_and_configs = llms_and_configs if llms_and_configs else DEFAULT_LLM_CONFIGS
        for k, v in llms_and_configs.items():
            llm = LLM_ENDPOINTS[k](**v)
            output = self.generate_and_execute(prompt=prompt, llm=llm, library=library, get_trace_url=get_trace_url, get_traceback=get_traceback)
            print("output from generate_and_execute: ", output)
            if not output['err']:
                output['llm_used'] = k
                return output
            print("error: ", output['err'])
        pass

    def get_default_params(self):
        return {
            'Chart Selector Template':self.get_chart_selector_prompt.template,
            'Step Retrieval Template':self.get_steps_prompt.template,
            'Code Generation Template':self.get_code_prompt.template,
            # 'Vars':", ".join(self.get_code_prompt.input_variables + self.get_steps_prompt.input_variables)
        }

    def init_chain(self, llm=None, retriever=None, steps_prompt=None, code_prompt=None, chart_selector_prompt = None):
        llm = llm if llm else self.llm #if llm is None else llm
        retriever = retriever if retriever else self.retriever
        self.prompts = []

        self.get_chart_selector_prompt = chart_selector_prompt or  self.get_chart_selector_prompt or PromptTemplate.from_template(
"You are a master chart visualizer tasked with determining the right type of plot for a given task. `df` is a pandas dataframe with the following columns and corresponding datatypes: {columns_and_types}. \
Given these chart-selection guidelines: `{chart_picking_guidelines}` and the user's intention: `{question}`, \
output an outline of your decision-making process, followed by the type of chart that would be best suited for the task. Delimit the chart-type in triple back-ticks. \
Pick from one of the following chart-types: `{chart_types}`"
        )
        self.prompts.append(self.get_chart_selector_prompt)

        self.get_steps_prompt = steps_prompt or self.get_steps_prompt or PromptTemplate.from_template("""`df` is a pandas dataframe with the following {num_cols} columns and corresponding datatypes: \
{columns_and_types}. Given these guidelines: `{guidelines}` and the user's intention: `{question}`, \
output a list of valid data processing and visualization steps to perform to `df` (using pandas and \
{library} only) towards producing a single chart. You may not change existing column names of `df`, but you can rename elements of the \
chart (such as axis titles) for clarity. Avoid complex transformations. Explain how each guideline is being met. """
        ) #if steps_prompt is None else steps_prompt
        # self.prompts.append({'name':get_code_prompt.run_name, 'item':get_code_prompt})
        self.prompts.append(self.get_steps_prompt)

        self.get_code_prompt = code_prompt or  self.get_code_prompt or PromptTemplate.from_template(
"`df` is a pandas dataframe with the following columns and corresponding datatypes: {columns_and_types}. \
Given these steps: {steps}, generate valid python code using the {library} library \
to address the user's question: `{question}`, producing a single {library} Figure at the end named `fig`. \
Assume `df` has already been defined. Use simple operations (no pivot). Delimit the code in triple backticks."
        )
        self.prompts.append(self.get_code_prompt)

        def invoke_lm(prompt, max_toks=600):
            resp = llm.invoke(prompt, max_tokens=max_toks)
            if hasattr(resp, 'content'):
                resp = resp.content
            return resp


        def get_steps(inputs):
            prompt = self.get_steps_prompt.invoke(inputs)
            # response = llm.invoke(prompt)#, max_tokens=10)
            response = invoke_lm(prompt)
            print("response from step genL ", response)
            response = response
            self.history.append({'input':inputs, 'intermediate':prompt, 'output':response, 'name':"get_steps_prompt"})
            return response
        
        
        def get_chart_type(inputs):
            prompt = self.get_chart_selector_prompt.invoke(inputs)
            chart_type = invoke_lm(prompt=prompt, max_toks=400)#llm.invoke(prompt, max_tokens=400)
            
            print("response from llm for chart type: ", chart_type)
            chart_type = self._clean_response(chart_type)
            print("cleaned: ", chart_type)
            
            self.history.append({'input':inputs, 'intermediate':prompt, 'output':chart_type, 'name':"get_steps_prompt"})
            return chart_type
        
        def get_code(inputs):
            prompt = self.get_code_prompt.invoke(inputs)
            response = llm.invoke(prompt)
            self.history.append({'input':inputs, 'intermediate':prompt, 'output':response, 'name':"get_code_prompt"})
            print("response from get_code: ", response)
            print(type(response))
            return response
        
        def retrieve_guidelines(inputs):
            json_steps_docs = retriever.invoke(inputs['chart_type'])
            formatted_guidelines = format_returned_guidelines_list(json_steps_docs, element_dict_transforms=lambda x : str(x))
            self.history.append({'input':inputs, 'guidelines':formatted_guidelines})
            print("retrieved steps: ", formatted_guidelines)
            return formatted_guidelines
        
        def get_chart_selection_guidelines(inputs):
            json_steps_docs = retriever.invoke(inputs['question'])
            formatted_guidelines = format_returned_guidelines_list(json_steps_docs, element_dict_transforms=lambda x : str(x))
            self.history.append({'input':inputs, 'guidelines':formatted_guidelines})
            print("retrieved steps: ", formatted_guidelines)
            return formatted_guidelines
        
        chain = (
            RunnablePassthrough.assign(chart_picking_guidelines=get_chart_selection_guidelines)
            | RunnablePassthrough.assign(chart_type=get_chart_type) 
            | RunnablePassthrough.assign(guidelines=retrieve_guidelines) 
            | RunnablePassthrough.assign(steps=get_steps)
            | RunnableLambda(get_code)
            | StrOutputParser()
        )
        return chain
    
    def init_debug_chain(self, llm=None):
        llm = llm if llm else self.debug_llm
        get_debug_prompt_traceback = PromptTemplate.from_template(
        """You are a python code fixer. Here is some buggy python code: {prev_code}. \
Here is traceback from the error: {traceback}. \
Your task is to do the following:
1. Find the bug
2. Explain what is causing the bug
3. Suggest steps to fix the bug. 
4. Finally, fix the bug and output valid python code that achieves the desired task. Delimit the code in triple back-ticks."""
        )

        chain = (
            get_debug_prompt_traceback
            | llm
            | StrOutputParser()
        )
        return chain

    
    # def modify_prompts()
    def set_generator_prompts(self, step_gen_prompt:Optional[str|PromptTemplate], code_gen_prompt:Optional[str|PromptTemplate]):
        step_gen_prompt = PromptTemplate.from_template(step_gen_prompt) if isinstance(step_gen_prompt, str) else step_gen_prompt
        code_gen_prompt = PromptTemplate.from_template(code_gen_prompt) if isinstance(code_gen_prompt, str) else code_gen_prompt

        # Assert variables are the same
        assert set(step_gen_prompt.input_variables) == set(self.get_steps_prompt.input_variables), "step generator prompt variables differ"
        assert set(code_gen_prompt.input_variables) == set(self.get_code_prompt.input_variables), "code generator prompt variables differ"

        self.chain = self.init_chain(steps_prompt=step_gen_prompt, code_prompt=code_gen_prompt)

    def set_chart_types(self, chart_types):
        self.chart_types = chart_types
        self.chain = self.init_chain()

    def set_main_llm(self, llm=None, llm_config=None, endpoint_type=HuggingFaceEndpoint):
        assert llm or (llm_config and endpoint_type), "either llm or llm_config must be set"

        self.llm = llm if llm else endpoint_type(**llm_config)

        if not self.debug_llm:
            self.debug_llm = self.llm
            self.debug_chain = self.init_debug_chain()

        self.chain = self.init_chain()

    def set_main_retriever(self, retriever=None):
        self.retriever = retriever
        self.chain = self.init_chain()

    def set_df_from_csv_str(self, csv_string):
        from io import StringIO
        self.set_df(pd.read_csv(StringIO(csv_string)))

    def set_df_from_csv_file(self, csv_filename):
        self.set_df(pd.read_csv(csv_filename))

    def set_df(self, df:pd.DataFrame):
        self.df=df
    
    @classmethod
    def _clean_response(cls, response:str, start='```', end='```'):
        match = re.search(rf'{start}(python)?(?P<code>.*?){end}', response, re.DOTALL)
        return match.group('code').strip() if match else response
    
    @classmethod
    def get_llm(cls, llm_name:str='mistral_7b', llm_config:dict=None):

        llm_config = llm_config if llm_config else DEFAULT_LLM_CONFIGS.get(llm_name, None)

        if llm_config is None:
            raise ValueError('No llm_config could be retrieved (if you\'re relying on defaults, the llm_name might be invalid).')

        llm = LLM_ENDPOINTS[llm_name](**llm_config)

        return llm
    

# print("openai token: ", OPENAI_TOKEN)
# print("debug mode: ", DEBUG)
    