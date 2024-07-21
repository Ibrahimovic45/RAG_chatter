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
from tempfile import NamedTemporaryFile
import re
import random


store = {}
session_id = str(random.randint(100,1000))

def read_pdf(file):
    #document = ""

    #reader = PdfReader(file)
    #for page in reader.pages:
      #document += str(reader.get_page_number(page))
      #document += page.extract_text()




   
    #document = ""
    
    #reader = PdfReader(file)
    
    # Iterate through each page of the PDF
    #for page_num in range(len(reader.pages)):
        #page = reader.pages[page_num]  # Get a specific page by index
        
        # Attempt to extract text from the page
        #page_text = page.extract_text()
        #if page_text:   # If text extraction is successful
            #page_text = page_text.strip()  # Remove leading/trailing whitespaces
            #document += f"Page {page_num + 1}:\n{page_text}\n\n"
        #else:
            #document += f"Page {page_num + 1} has no text.\n\n"
    
    
    #doc = []
    #if files:
      #temp_file = "./temp.pdf"
      #with open(temp_file, "wb") as file:
        #file.write(files.read())
        #file_name = files.name
      #loader = PyMuPDFLoader(temp_file)
      #doc.extend(loader.load())

    #docs = []
    bytes_data = file.read()
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
      tmp.write(bytes_data)                      # write data from the uploaded file into it
      document = PyPDFLoader(tmp.name).load_and_split()        # <---- now it works!
    os.remove(tmp.name)                            # remove temp file

    #with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
      #f.write(file.getbuffer())
      #l#oader = PyPDFLoader(f.name)
      #document = loader.load()
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


#@st.cache_resource
#def Instructor():
  #instructor_embeddings = HuggingFaceInstructEmbeddings(
            #model_name=model_name, 
        #) #, model_kwargs={"device":"cuda"}

  #instructor_embeddings = Replicate(
   #model = "replicate/all-mpnet-base-v2:b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
    #)

  #return instructor_embeddings

def retriever(existing_vector_store, instructor_embeddings):
    load_db = FAISS.load_local(
                "vector store/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
    )
    return load_db

def embedding_storing(model_name, split, create_new_vs, new_vs_name):
    
    if create_new_vs is not None:
        # Load embeddings instructor
        instructor_embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name, 
        ) #, model_kwargs={"device":"cuda"}
        #instructor_embeddings = Instructor()

        # Implement embeddings
        db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs == True:
            # Save db
            db.save_local("vector store/" + new_vs_name)
        else:
            # Load existing db
            load_db = retriever()
            # Merge two DBs and save
            load_db.merge_from(db)
            load_db.save_local("vector store/" + new_vs_name)

        st.success("The document has been saved.")




def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@st.cache_resource(ttl=300, max_entries=1)
def Llm():
  os.environ["GROQ_API_KEY"] = st.secrets["token"]
  llm = ChatGroq(model="llama3-8b-8192",
  temperature=0.7,
  max_tokens=2000
  )
  return llm


#### Tis function to cache
#@st.cache_resource
def prepare_rag_llm(
    llm_model, instruct_embeddings, vector_store_list, temperature, max_length
): 
    # Load embeddings instructor
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name=instruct_embeddings, 
    ) #, model_kwargs={"device":"cuda"}
    #instructor_embeddings = Instructor()

    # Load db
    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}", instructor_embeddings, allow_dangerous_deserialization=True
    )

    # Load LLM
    #llm = HuggingFaceHub(
        #repo_id=llm_model,
        #model_kwargs={"temperature": temperature, "max_length": max_length},
        #huggingfacehub_api_token=token
    #)


    #os.environ["GROQ_API_KEY"] = "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"
#

    #llm = ChatGroq(model="llama3-8b-8192",
    #temperature=0.7,
    #max_tokens=2000
    #)
    llm = Llm()

    #memory = ConversationBufferWindowMemory(
        #k=2,
        #memory_key="chat_history",
        #output_key="answer",
        #return_messages=True,
    #)
    #return llm, loaded_db

#def create_chatbot(llm, loaded_db):
    #memory = ConversationBufferWindowMemory(
        #k=1,
        #memory_key="chat_history",
        #output_key="answer",
        #return_messages=True,
    #)
    # Create the chatbot
    #qa_conversation = ConversationalRetrievalChain.from_llm(
        #llm=llm,
        #chain_type="stuff",
        #retriever=loaded_db.as_retriever(),
        #return_source_documents=True,
       # memory=memory,
    #)
    
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
        llm, loaded_db.as_retriever(), contextualize_q_prompt
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


    ### Statefully manage chat history ###
    store = {}

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

def generate_answer(question):
    response = st.session_state.conversation.invoke({"input": question},
    config={
    "configurable": {"session_id": session_id}})
    answer = response["answer"]
    explanation = response["context"]
    doc_source = {}
    for d in explanation:
      doc_source["Page " + str(d.metadata["page"])] = [d.page_content, d.metadata["source"]]
    
    return answer, doc_source 
