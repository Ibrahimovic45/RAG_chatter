import streamlit as st
import os
from pages.backend import rag_functions
import random
from google.cloud.storage import Client, transfer_manager
import shutil
import atexit


if "session_id" in st.session_state:
     session_id_saved = st.session_state.session_id
else:
    session_id_saved = ""

st.cache_data.clear()
rag_functions.retriever.clear()

# Create this folder locally
if not os.path.exists("data"):
  os.makedirs("data")


bucket, blob_names, new_blob_names   = rag_functions.cloud()

st.title("Chatbot")
 
with st.expander("Username and sources"):
  row_1 = st.columns(2)
  with row_1[0]:
            
            vector_store_list = list(set(new_blob_names)) + [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
            #vector_store_list = os.listdir("vector store")
            default_choice = (
                vector_store_list.index('Harry_Potter_1') #Harry_Potter_1
                if 'Harry_Potter_1' in vector_store_list
                else 0
            )

            existing_vector_store = st.selectbox("Vector Store", vector_store_list, default_choice)
           
  with row_1[0]:
            if session_id_saved != "":
                value = session_id_saved
            else:
                value = ""
            session_id = st.text_input("Username", value=value) 
            os.environ["session_id"] = session_id    
              

# Setting the LLM
llm_model = "meta-llama/Meta-Llama-3-8B-Instruct"
instruct_embeddings = "Lajavaness/bilingual-embedding-large" #"Alibaba-NLP/gte-Qwen2-1.5B-instruct"  #"sentence-transformers/all-mpnet-base-v2" #hkunlp/instructor-xl


# Prepare the LLM model
if "conversation" not in st.session_state:
    st.session_state.conversation = None

if session_id:
    #if not os.path.exists("data/" + existing_vector_store):
      #os.makedirs("data/" + existing_vector_store)
    rag_functions.downloader(bucket, blob_names, existing_vector_store)

    st.session_state.session_id = session_id
    st.session_state.conversation = rag_functions.prepare_rag_llm( llm_model, 
    instruct_embeddings, existing_vector_store) #removed , temperature, max_length


# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Source documents
if "source" not in st.session_state:
    st.session_state.source = []

# Display chats
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ask a question
if question := st.chat_input("Ask a question"):
    # Append user question to history
    st.session_state.history.append({"role": "user", "content": question})
    # Add user question
    with st.chat_message("user"):
        st.markdown(question)

    # Answer the question
    answer, doc_source = rag_functions.generate_answer(question, session_id)
    
    new_doc_source = {}
    for key,content in doc_source.items():
      if len(content[0]) > 100:
          truncated_text = content[0][:200] + "..."
      else:
          truncated_text = content[0]
      new_doc_source[key] = truncated_text

    with st.chat_message("assistant"):
        st.write(answer)
    # Append assistant answer to history
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Append the document sources
    st.session_state.source.append({"question": question, "answer": answer, "document": new_doc_source})


# Source documents
with st.expander("Source documents"):
    st.write(st.session_state.source)

#rag_functions.cleanup()

