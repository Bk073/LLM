## Environment setup:
    - python3 -m venv .env
    - source .env\bin\activate -> to activate environment
    - deactivate -> to deactivate environment

    - manually set key in os.environ["OPENAI_API_KEY"] in utils.py 
    - to get the openAPI key goto https://openai.com/blog/openai-api


## Installation
    - pip install -r requirements.txt

## Running inference
### web app
    - python3 app.py
    - run /predict endpoint for prediction
    - send a post request with file_path to pdf and question
