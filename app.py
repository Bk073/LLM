from flask import Flask, request
from flask_cors import CORS
import json
from langchain_qa import *
from utils import *
from prepareData import load_vectordb

app = Flask(__name__)

CORS(app)

@app.route("/")
def hello():
    return "<h1>Welcome to LLM project</h1>"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        persist_directory = './docs/chroma/'
        # vectordb = prepare_data(data['file_path'], persist_directory, clear_dir=True)
        vectordb = load_vectordb(persist_directory, embed='huggingface')
        if data['model'] == 'openai':
            llm = get_gpt_llm()
        elif data['model']== 'local':
            llm = get_local_llm()
        result = get_prediction(llm, vectordb, data['question'], data['model'])
        print(result)
        return json.dumps({"answer": result})

if __name__ == "__main__":
    setup()
    app.run(debug=True, port=8082)