from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from utils import setup
import os
import shutil
import argparse

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits

def save_to_db(embed, splits, persist_directory, clear_dir=False):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    if clear_dir:
        shutil.rmtree(persist_directory)
    if embed=='openai':
        print("open ai embedding ***********")
        embeddings = OpenAIEmbeddings()
    elif embed=='huggingface':
        print("huggingface embedding ***********")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
        )
    vectordb.persist()
    return True

def prepare_data(embed, file_path, persist_directory, clear_dir):
    if isinstance(file_path, list):
        for file_pth in file_path:
            docs = load_document(file_pth)
            splits = split_docs(docs)
            save_to_db(embed, splits, persist_directory, clear_dir)
    else:
        docs = load_document(file_path)
        splits = split_docs(docs)
        save_to_db(embed, splits, persist_directory, clear_dir)

    return True

def load_vectordb(persist_directory, embed):
    if embed=='openai':
        embeddings = OpenAIEmbeddings()
    elif embed=='huggingface':
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pdf files")
    parser.add_argument("--filepath", type=str, help="path to pdf files")
    parser.add_argument("--dir", type=str, help="location to store embedding vector", default='./docs/chroma/')
    parser.add_argument("--embed", type=str, help="embedding to be used", default='huggingface')
    parser.add_argument("--clear_dir", type=str, help="Set true if you want to clear the directory", default=False)

    args = parser.parse_args()
    setup()
    success = prepare_data(args.embed, args.filepath, args.dir, args.clear_dir)
    if success:
        print(f"Succeccfully saved {args.filepath.split('/')[-1]}")
    # example:
    # python3 prepareData.py --filepath "/Users/bishwakarki/Desktop/research/watermarking/Xin_Zhong/Image_Watermarking/one.pdf"
    #  --embed "openai" --clear_dir "False"

