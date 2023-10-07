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

def create_db_from_directory(dir_path: str) -> dict:
    pdf_files = [f for f in os.listdir(dir_path) if f.endswith('.pdf')]
    db_dict = {}
    
    for pdf_file in pdf_files:
        full_pdf_path = os.path.join(dir_path, pdf_file)
        with open(full_pdf_path, "rb") as pdf:
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
 
            store_name = pdf_file[:-4]
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                    
            db_dict[store_name] = VectorStore
            
    return db_dict

def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="gpt-3.5-turbo")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant for finding materials.
        
        Answer the following question: {question}
        By searching the following source documentation: {docs}
        
        Only use the factual information from the documents to answer the question.
        
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
    
    dir_path = st.text_input("Enter directory path containing PDFs")
    
    if dir_path:
        db_dict = create_db_from_directory(dir_path)
        selected_pdf = st.selectbox('Select a PDF or "All PDFs"', list(db_dict.keys()) + ["All PDFs"])
        query = st.text_input("Ask a question")
        if query:
            if selected_pdf == "All PDFs":
                combined_response = ""
                for db_name, db in db_dict.items():
                    response, docs = get_response_from_query(db, query)
                    print(response, docs)
                    combined_response += f"For {db_name}: {response}\n\n"
                    for doc in docs:
                        combined_response += str(doc) + "\n"
                st.write(combined_response)
            else:
                db = db_dict[selected_pdf]
                response, docs = get_response_from_query(db, query)
                st.write(response)
                for doc in docs:
                    st.write(str(doc))
                    st.write()

if __name__ == "__main__":
    main()
