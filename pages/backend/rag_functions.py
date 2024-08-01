import streamlit as st
from langchain.document_loaders import TextLoader
from pypdf import PdfReader
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
#from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import getpass
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import tempfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
import re
import random
from st_files_connection import FilesConnection
from google.cloud import storage
from google.cloud.storage import Client, transfer_manager
from pathlib import Path
import shutil

store = {}

def cleanup():
    #global os.path("data")
    if os.path.exists("data"):
        shutil.rmtree("data")

def cloud():
  storage_client = Client()
  bucket = storage_client.bucket("rag_chat_bucket_2")
  blob_names = [blob.name for blob in bucket.list_blobs()]
  new_blob_names = []
  for name in blob_names:
      new_name = name.split('/')[-2] 
      new_blob_names.append(new_name)
  return bucket, blob_names, new_blob_names    


def read_pdf(file):
    bytes_data = file.read()
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
      tmp.write(bytes_data)                      # write data from the uploaded file into it
      document = PyPDFLoader(tmp.name).load_and_split()        # <---- now it works!
    os.remove(tmp.name)                            # remove temp file
    return document


def read_txt(file):
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")

    return document


def split_doc(document, chunk_size, chunk_overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)

    return split

@st.cache_data(ttl=60, max_entries=4, show_spinner="Downloading data...")
def downloader(_bucket, blob_names, existing_vector_store):    

  #new_blob_names = []

  for name in blob_names:
    new_blob_name = name.split('/')[-1]
    if name.split('/')[-2] == existing_vector_store:
      blob = _bucket.blob(name)
      blob.download_to_filename("data" + f"/{new_blob_name}")
      

@st.cache_data(ttl=60, max_entries=1, show_spinner="Loading data...")
def retriever(existing_vector_store, _instructor_embeddings):
    
  
  load_db = FAISS.load_local(f"vector store/{existing_vector_store}",
    _instructor_embeddings, allow_dangerous_deserialization=True
          )
  
  
  return load_db


def embedding_storing(model_name, split, create_new_vs, existing_vector_store, new_vs_name):
    
    if create_new_vs is not None:
        # Load embeddings instructor
        instructor_embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name, 
        ) #model_kwargs={"device":"cuda"}

        # Implement embeddings
        st.session_state.db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs == True:
            # Save db
            st.session_state.db.save_local("vector store/" + new_vs_name)                      
            
        else:
            # Load existing db
            st.session_state.load_db = retriever(existing_vector_store, instructor_embeddings)
            # Merge two DBs and save
            st.session_state.load_db.merge_from(st.session_state.db)
            st.session_state.load_db.save_local("vector store/" + new_vs_name)

        st.success("The document has been saved.")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@st.cache_resource(ttl=10, max_entries=1, show_spinner="Loading model...")
def Llm():
  os.environ["GROQ_API_KEY"] = st.secrets["token"]
  llm = ChatGroq(model="llama3-8b-8192",
  temperature=0.7,
  max_tokens=2000
  )
  return llm


#@st.cache_resource
def prepare_rag_llm(
    llm_model, instruct_embeddings, vector_store_list
): # removed temperature, max_length
    # Load embeddings instructor
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name=instruct_embeddings, 
    )  #model_kwargs={"device":"cuda"}
    st.session_state.loaded_db = retriever(vector_store_list, instructor_embeddings)


    # Load LLM
    llm = Llm()   
    
    # With history
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.loaded_db.as_retriever(), contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the "
        "answer concise and answer in the same language as the question."
        "Include page numbers of the retrieved context at the end of the answer"
         "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return rag_chain


def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

def generate_answer(question, session_id):
  answer = "An error has occured"

  if session_id == "":
        answer = "Please insert your name"
        doc_source = {"Page":"no source"}
  else:
    response = st.session_state.conversation.invoke({"input": question},
    config={
    "configurable": {"session_id": session_id}})
    answer = response["answer"]
    explanation = response["context"]
    doc_source = {}
    for d in explanation:
      doc_source["Page " + str(d.metadata["page"])] = [d.page_content, d.metadata["source"]]
    
  return answer, doc_source 
