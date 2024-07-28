import streamlit as st
from dotenv import load_dotenv
from utils import get_answer, isUrl, llm, generate_answerbook, chain

import os
if 'urls' not in st.session_state:
  st.session_state.urls = [""] * 3
load_dotenv()
st.title("AutoSage 📃🔎")
st.sidebar.title("Enter Your Urls here")

main_placeholder = st.empty()
status_placeholder = st.empty()
result_placeholder = st.empty()
source_placeholder = st.empty()

for i in range(3):
  st.session_state.urls[i] = st.sidebar.text_input(f"paste url-{i+1} here")
button_clicked = st.sidebar.button("Generate Answerbook")

if button_clicked:
  print("clicked",st.session_state.urls )
  urls_from_user= [url for url in st.session_state.urls if isUrl(url)]
  if len(urls_from_user) !=0:
    print(urls_from_user)
    
    st.session_state.db = generate_answerbook(urls=urls_from_user,element= main_placeholder)
    st.header("Answer Book Generated✔️✔️")
    


st.session_state.query = main_placeholder.text_input("Enter Your Query Here And Press Enter")


if st.session_state.query !="" and "db" in st.session_state:
    
    results = get_answer(st.session_state.query, st.session_state.db, status_placeholder)
    print(results)
    result_placeholder.subheader(results["result"])
    source_placeholder.markdown(f"[{results['source']}]({results['source']})")
    
if __name__ == "__main__":
  pass
  # book = generate_answerbook(urls=[ "https://en.wikipedia.org/wiki/Virat_Kohli"])
  # query= "what is age of virat kohli"
  # print(get_answer(query, book))
  
  
