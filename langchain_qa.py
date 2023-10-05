import openai
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import shutil

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits

def save_to_db(splits, persist_directory, clear_dir=True):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    if clear_dir:
        shutil.rmtree(persist_directory)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def prepare_data(file_path, persist_directory, clear_dir):
    docs = load_document(file_path)
    splits = split_docs(docs)
    vectordb = save_to_db(splits, persist_directory, clear_dir)
    return vectordb


def get_gpt_llm():
    llm_name='gpt-3.5-turbo'
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    return llm

def get_prediction(llm, vectordb, question):
    template_2 = """Use the following pieces of context to answer the question at the end.
        Answer the question in detail and as concise as possible. Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Helpful Answer:"""

    QA_CHAIN_PROMPT_2 = PromptTemplate.from_template(template_2)
    qa_chain_2 = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_2}
    )
    result_p_2= qa_chain_2({"query":question})
    return result_p_2['result']