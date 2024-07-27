from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings
import re

warnings.filterwarnings("ignore")


load_dotenv()


llm = GoogleGenerativeAI(model="models/text-bison-001",
                         google_api_key=os.environ.get("GOOGLE_API_KEY"),
                         temperature=0.1)

template = PromptTemplate(
  input_variables=["query", "dataset"],
  template = 'for the {query}. Please give me the most relevent result and source link from one of these given {dataset}. give results in xml format {{result, source}} field'
)

chain = LLMChain(llm = llm, prompt=template, output_key = "result")

def generate_answerbook(urls):
  loader = UnstructuredURLLoader(urls=urls)
  data = loader.load()

  splitter = RecursiveCharacterTextSplitter(
      separators=['\n', '.', " "],
      chunk_size=200,
      chunk_overlap=0,
  )

  doc = splitter.split_documents(data)

  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2")

  index = FAISS.from_documents(doc, embeddings)

  return index


def get_answer(query, db):
  try:
    res = chain.invoke({"query":query, "dataset" :db.similarity_search(query, k=4)})
    result = re.search(r'<result>(.*?)</result>', res["result"]).group(1)
    source = re.search(r'<source>(.*?)</source>', res["result"]).group(1)
    
    return {"result": "result", source:source }

  except Exception as e:
    print(e)
    return {"result":" ", "source":" "}
  

if __name__ == "__main__":
  print(llm.invoke("who is virat kohli tell in 10 words"))
