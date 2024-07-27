import streamlit as st
from dotenv import load_dotenv
from utils import get_answer, llm, generate_answerbook, chain
import warnings

# warnings.filters["all"]
warnings.filterwarnings("ignore")
import os

load_dotenv()

if __name__ == "__main__":
  # print(llm.invoke("who is virat kohli tell in 10 words"))
  book = generate_answerbook(urls=[ "https://en.wikipedia.org/wiki/Virat_Kohli"])
  query= "what is age of virat kohli"
  print(get_answer(query, book))
  
  
