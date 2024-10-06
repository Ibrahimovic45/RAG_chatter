import streamlit as st
import os
from pages.backend import rag_functions


bucket, blob_names, new_blob_names   = rag_functions.cloud() 


st.title("Document Uploading")
st.markdown("This page is used to upload the documents as the custom knowledge for the chatbot.")

with st.form("document_input"):
    
    document = st.file_uploader("Knowledge Documents", type=['pdf', 'txt'], help=".pdf or .txt file, less than 20MB preferred ")

    row_1 = st.columns([2, 1, 1])
    with row_1[0]:
        #instruct_embeddings = st.text_input(
            #"Model Name of the Instruct Embeddings", value="sentence-transformers/all-mpnet-base-v2"
        #)
        st.write("Embedding model")
        st.write("sentence-transformers/all-mpnet-base-v2")
        instruct_embeddings = "Lajavaness/bilingual-embedding-large" #"sentence-transformers/all-mpnet-base-v2"
    
    with row_1[1]:
        #chunk_size = st.number_input(
            #"Chunk Size", value=200, min_value=0, step=1,
        #)
        chunk_size = 200
        st.write("Chunk size")
        st.write("200")
    
    with row_1[2]:
        #chunk_overlap = st.number_input(
            #"Chunk Overlap", value=10, min_value=0, step=1, help="higher that chunk size"
        #)
        chunk_overlap = 10
        st.write("Chunk Overlap")
        st.write("10")
    
    row_2 = st.columns(2)
    with row_2[0]:
        # List the existing vector stores
        vector_store_list = list(set(new_blob_names))
        vector_store_list = ["<New>"] + vector_store_list
        #vector_store_list = os.listdir("vector store/")
        #vector_store_list = ["<New>"] + vector_store_list
        existing_vector_store = st.selectbox(
            "Document to Merge the Knowledge with", vector_store_list,
            help="Which document to add the new documents. Choose <New> to create a new document."
        )

    with row_2[1]:
        # List the existing vector stores     
        new_vs_name = st.text_input(
            "New Document Name", value="new_vector_store_name",
            help="If choose <New> in the dropdown / multiselect box, name the new document. Otherwise, fill in the existing document to merge with."
        )

    save_button = st.form_submit_button("Save vector store")

if save_button:
    
    # Read the uploaded file
    if document.name[-4:] == ".pdf":
        document1 = rag_functions.read_pdf(document)
    elif document.name[-4:] == ".txt":
        document = rag_functions.read_txt(document)
    else:
        st.error("Check if the uploaded file is .pdf or .txt")

    # Split document
    if document.name[-4:] == ".pdf":
      split = document1 #in place of just document1
    else:
      split = rag_functions.split_doc(document, chunk_size, chunk_overlap)


    # Check whether to create new vector store
    create_new_vs = None
    if existing_vector_store == "<New>" and new_vs_name != "":
        create_new_vs = True
    elif existing_vector_store != "<New>" and new_vs_name != "":
        create_new_vs = False
    else:
        st.error("Check the 'Vector Store to Merge the Knowledge' and 'New Vector Store Name'")
    
    # Embeddings and storing
    rag_functions.downloader(bucket, blob_names, existing_vector_store)

    rag_functions.embedding_storing(
        instruct_embeddings, split, create_new_vs, 
        existing_vector_store,new_vs_name
    )
