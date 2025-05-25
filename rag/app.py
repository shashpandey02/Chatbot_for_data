import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os

st.title("📊 Telemetry Equipment Assistant")

# file upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    # Load and display the dataframe
    df = pd.read_excel(uploaded_file)

    # Convert to LangChain documents
    docs = [
        Document(page_content=row.to_string(), metadata={"row_index": idx})
        for idx, row in df.iterrows()
    ]

    ## divide the entire document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    # embedding the data and storing in Vector store
    embedding = HuggingFaceBgeEmbeddings(model_name = "all-MiniLM-L6-v2") #Fast and Lightweight
    db = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="chroma_db")

    llm  = Ollama(model = "llama3.2")
    ## Design ChatPrompt Template
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant that helps users explore and manage equipment information from an inventory dataset. 
        Answer questions about specific equipment, summarize failure counts, identify equipment by location, or list items by attributes such as type or ID.
        I want to know about [specific detail or ID]. 
        (Examples: 
        - "List all equipment in TELEMETRY with more than 1 failure."
        - "What is the location and service number of NOP/KVM/01?"
        - "Summarize all DELL-T5600 computers and their status.")
        You can refer to equipment by:
        - ID (eqID)
        - Name (eqName)
        - Location (loc)
        - Failure count (fails)
        - Service number (srvNo)
        <context>
        {context}
        </context>
        Question : {input} """)
    ## Chain Introduction
    ## Create Stuff Document Chain

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    query = st.text_input("Ask a question about your equipment data:")
    
    if query:
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input":query})
            result = response['answer']
        st.success('Answer :')
        st.write(result)



    


