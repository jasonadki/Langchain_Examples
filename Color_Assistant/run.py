from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv, find_dotenv
import textwrap

import streamlit as st
from PyPDF2 import PdfReader

import os
import pickle


import json


load_dotenv(find_dotenv())



def create_db_from_json(jsonFile: str) -> FAISS: 
    if jsonFile is not None:

        print(jsonFile)

        # load the json UploadedFile(id=1, name='Color_Coding_1_decomposed.json', type='application/json', size=15514)
        jsonFile = json.load(jsonFile)
        

        # Extract the list of dictionaries
        info_list = jsonFile['Information']

        # Convert each dictionary into a string (JSON format)
        texts = [json.dumps(info_dict) for info_dict in info_list]
        metadatas = [info_dict for info_dict in info_list]

        # Generate embeddings
        embeddings = OpenAIEmbeddings()
        text_embeddings = embeddings.embed_documents(texts)

        # Create pairs of (text, embedding)
        text_embedding_pairs = list(zip(texts, text_embeddings))

        # Create the FAISS index
        faiss_index = FAISS.from_embeddings(text_embedding_pairs, embeddings, metadatas=metadatas)
                        
        return faiss_index


def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about color codings for information.
        
        Answer the following question: {question}
        By searching the following color coding guide: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    

    # for doc in docs:
    #     print(str(doc.metadata["page"]) + ":", doc.page_content[:800])
    #     print()

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


def main():
    st.title("Proposal Assistant")
    
    pdf = st.file_uploader("Upload a PDF", type="json")
    
    if pdf is not None:
        db = create_db_from_json(pdf)
        query = st.text_input("Ask a question")
        if query:
            response, docs = get_response_from_query(db, query)
            # str(doc.metadata["page"]) + ":", doc.page_content[:800]
            st.write(response)
            for doc in docs:
                st.write(str(doc))
                st.write()
            # st.write(docs)
            


if __name__ == "__main__":
    # # Example usage:
    # pdf_url = "Resources/Halcyon - Aeolus.pdf"
    # db = create_db_from_pdf(pdf_url)

    # query = "How much will the proposal cost?"
    # response, docs = get_response_from_query(db, query)
    # print(response, docs)
    
    main()