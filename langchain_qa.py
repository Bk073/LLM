import openai
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp  
from langchain.callbacks.manager import CallbackManager  
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
import os
import shutil

def get_gpt_llm():
    llm_name='gpt-3.5-turbo'
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    return llm

def get_local_llm():
    # pip install llama-cpp-python
    model_path = "./llama-2-70b.Q4_K_M.gguf"
    # n_gpu_layers = 0 # Metal set to 1 is enough.  
    n_batch = 512 # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.   
    llm = LlamaCpp(  
        model_path=model_path,  
        # n_gpu_layers=n_gpu_layers,  
        n_batch=n_batch,  
        n_ctx=4650,  
        f16_kv=True, # MUST set to True, otherwise you will run into problem after a couple of calls  
        verbose=True,  
        temperature=0.4
        )  
    return llm

def get_prediction(llm, vectordb, question, model):
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
    if model=='openai':
        result_p_2= qa_chain_2({"query":question})
        return result_p_2['result']
    else:
        result_p_2 = qa_chain_2.run(question)
        return result_p_2
    