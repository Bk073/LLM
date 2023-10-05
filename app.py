from flask import Flask, request
from flask_cors import CORS
import json
from langchain_qa import *
from utils import *

app = Flask(__name__)

CORS(app)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print("request files", request.get_json())
        data = request.get_json()
        persist_directory = 'docs/chroma/'
        vectordb = prepare_data(data['file_path'], persist_directory, clear_dir=True)
        llm = get_gpt_llm()
        result = get_prediction(llm, vectordb, data['question'])
        print(result)
        return json.dumps({"answer": result})

if __name__ == "__main__":
    setup()
    app.run(debug=True)