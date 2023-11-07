import sys
import os
import openai
from dotenv import load_dotenv, find_dotenv

def setup():
    sys.path.append('./content')
    os.environ["OPENAI_API_KEY"] = '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'
    _ = load_dotenv(find_dotenv())
    openai.api_key  = os.environ['OPENAI_API_KEY']