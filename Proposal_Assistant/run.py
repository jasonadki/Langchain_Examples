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





load_dotenv(find_dotenv())



def create_db_from_pdf(pdf: str) -> FAISS:
    # Get path to PDF
    
    
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
                
        return VectorStore


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
        You are a helpful assistant that that can answer questions about formal responses
        sent following a Request for Proposal.
        
        Answer the following question: {question}
        By searching the following PROPOSAL RESPONSE: {docs}
        
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
    
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    
    if pdf is not None:
        db = create_db_from_pdf(pdf)
        query = st.text_input("Ask a question")
        if query:
            response, docs = get_response_from_query(db, query)
            st.write(response, docs)
            # st.write(docs)
            


if __name__ == "__main__":
    # # Example usage:
    # pdf_url = "Resources/Halcyon - Aeolus.pdf"
    # db = create_db_from_pdf(pdf_url)

    # query = "How much will the proposal cost?"
    # response, docs = get_response_from_query(db, query)
    # print(response, docs)
    
    main()