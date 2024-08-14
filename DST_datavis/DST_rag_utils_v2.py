from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter, HTMLHeaderTextSplitter, HTMLSectionSplitter
import json
from langchain_chroma import Chroma
import os
from typing import Optional, List

DEBUG = os.getenv('DEBUG')


RETRIEVERS_DICT = {
    'chroma':Chroma,
}

class DummyRetriever(BaseRetriever):
    """Dummy retriever class that's initialized with a list of string chunks/documents and simply
    returns each of them when invoked (completely ignores query)"""
    k:int = 0
    documents: Optional[List[str]|List[Document]]
    """List of documents to retrieve from."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = [Document(page_content=strdoc) for strdoc in self.documents] if isinstance(self.documents, list) else \
        self.documents
        return docs

def _read_and_split_json(fname, max_chunk_size=10):
    json_splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
    try:
        with open(fname) as f:
            json_dict = json.load(f)
            split_json_chunks = json_splitter.split_text(json_data=json_dict)
            return split_json_chunks
    except Exception as e:
        print("DST_rag_utils: error reading file: ", str(e))

def _get_vec_db_from_persist(retriever_name='chroma', embeddings_api=HuggingFaceEmbeddings, embeddings_model_name="all-MiniLM-L6-v2", reset=False, persist_directory=os.getcwd(), overwrite=False):
    embedding_function = embeddings_api(model_name=embeddings_model_name)
    vec_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    return vec_db

def _get_vec_db(docs, retriever_name='chroma', embeddings_api=HuggingFaceEmbeddings, embeddings_model_name="all-MiniLM-L6-v2", reset=False, persist_directory=os.getcwd(), overwrite=False):
    embedding_function = embeddings_api(model_name=embeddings_model_name)
    vec_db = Chroma.from_texts(docs, embedding_function, persist_directory=persist_directory)
    return vec_db


def _get_retriever_persistent_client(fname, retriever_name='chroma', embeddings_api=HuggingFaceEmbeddings, embeddings_model_name="all-MiniLM-L6-v2", max_chunk_size=10, persist_directory=os.getcwd(), search_kwargs={'k':4}):
    persistent_client = chromadb.PersistentClient()
    persistent_client.get
    collection = persistent_client.get_or_create_collection("guidelines")
    # Chroma
    
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="collection_name",
        embedding_function=None#embedding_function,
    )

    return langchain_chroma

def get_langchain_retriever_from_json_file(fname, retriever_name='chroma', embeddings_api=HuggingFaceEmbeddings, embeddings_model_name="all-MiniLM-L6-v2", max_chunk_size=10, persist_directory=os.getcwd(), search_kwargs={'k':2}):
    from pathlib import Path
    if Path(f'{persist_directory}/chroma.sqlite3').exists():
        if DEBUG:
            print("db already exists")
        vec_db = _get_vec_db_from_persist(retriever_name=retriever_name, 
                                            embeddings_api=embeddings_api,
                                            embeddings_model_name=embeddings_model_name,
                                            persist_directory=persist_directory)
    else:
        if DEBUG:
            print("making new db")
            
        docs = _read_and_split_json(fname=fname, max_chunk_size=max_chunk_size)
        vec_db = _get_vec_db(retriever_name=retriever_name, 
                                            embeddings_api=embeddings_api,
                                            embeddings_model_name=embeddings_model_name,
                                            docs=docs,
                                            persist_directory=persist_directory)
    
    retriever = vec_db.as_retriever(search_kwargs=search_kwargs)
    
    return retriever

def format_returned_guidelines_list(docs, element_dict_transforms = lambda x:str(json.loads(x))):#lambda d : "\n".join(d['chart_types'].values())):
    result = [element_dict_transforms((doc.page_content)) for doc in docs]
    return result
