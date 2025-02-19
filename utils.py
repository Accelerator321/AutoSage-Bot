from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
import re
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = GoogleGenerativeAI(model="models/text-bison-001",
                         google_api_key=os.environ.get("GOOGLE_API_KEY"),
                         temperature=0.1)

template = PromptTemplate(
    input_variables=["query", "dataset"],
    template=
    'for the {query}. Please give me the most relevent result and source link from one of these given {dataset}. give results in xml format {{result, source}} field'
)

chain = LLMChain(llm=llm, prompt=template, output_key="result")


def generate_answerbook(urls, element):
  element.text("Loading Data...✔️✔️")
  loader = UnstructuredURLLoader(urls=urls)
  data = loader.load()

  splitter = RecursiveCharacterTextSplitter(
      separators=['\n', '.', " "],
      chunk_size=500,
      chunk_overlap=100,
  )
  element.text("Splitting Text...✔️✔️")
  doc = splitter.split_documents(data)
  element.text("Generating Embeddings..✔️✔️")
  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2")

  index = FAISS.from_documents(doc, embeddings)

  return index


def get_answer(query, db, element):
  try:
    element.subheader("Generating Answer.....✔️")
    res = chain.invoke({
        "query": query,
        "dataset": db.similarity_search(query, k=8)
    })

    element.subheader("Answer Generated ✔️✔️")
   
    result = re.search(r'<result>(.*?)</result>', res["result"]).group(1)
    source = re.search(r'<source>(.*?)</source>', res["result"]).group(1)

    return {"result": result, "source": source}

  except Exception as e:
    print(e)
    return {"result": " ", "source": " "}


def isUrl(link):
  regex= 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' 
  URL= re.findall(regex,link) 
  return len(URL)>0 
if __name__ == "__main__":
  print(llm.invoke("who is virat kohli tell in 10 words"))
