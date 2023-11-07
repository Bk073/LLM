import sys
import os
import openai
from dotenv import load_dotenv, find_dotenv

def setup():
    sys.path.append('./content')
    os.environ["OPENAI_API_KEY"] = '#######'
    _ = load_dotenv(find_dotenv())
    openai.api_key  = os.environ['OPENAI_API_KEY']
