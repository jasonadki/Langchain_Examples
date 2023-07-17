from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv

import os

import streamlit as st

import json

from docxtpl import DocxTemplate

def update_word_doc(template_path, context, output_path):
    doc = DocxTemplate(template_path)
    doc.render(context)
    doc.save(output_path)


def extract_name(path):
    base_name = os.path.basename(path)  # get the last part of the path

    # if this is a file name, remove the extension
    if '.' in base_name:
        base_name = os.path.splitext(base_name)[0]

    return base_name



def process_file(dirpath, file, embeddings):
    # Initialize list to store documents
    docs = []

    # print full path and filename starting from root
    full_name = os.path.join(dirpath, file)


    # Load the text of the file
    loader = TextLoader(full_name, encoding="utf-8")

    # Split the text into documents
    docs.extend(loader.load_and_split())


    # Chunk the files
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    
    # Store the embeddings in a vector store
    db = FAISS.from_documents(texts, embeddings)

    # Retrieve the embeddings from the vector store
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    # Load the model
    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k") 



    # Load the template that is prompting the agent
    template='''
            You are a code peer reviewer and a software security reviewer. 
            I want you to write me a summary of what the code does and whether there are any considerable
            security concerns within the code that would cause harm to my computer or the network my computer is on.
            It is crucial that your responses include all fields below, leaving none missing. The format of your response should be a python dictionary like the following:

            {
            "Summary": <Short summary of what the code does>
            "Security_Concerns": <Whether or not there are security concerns and if there are what are they>
            "Security_Rating": <A security rating from the following: [Very Low, Low, Moderate, High, Very High]>
            "Justification": <Why you gave it the security rating chosen>
            }

            Remember, it's absolutely critical to include all fields in your response and in the dictionary format.
            '''

    # Create a Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever
    )

    # Create a list to store the chat history 
    chat_history = []

    # Generate a question and response from the model
    result = qa({"question": template, "chat_history": chat_history})


    result_dict = json.loads(result['answer'])
    result_dict["Name"] = full_name

    
    
    # Append the result to the results.json file
    with open("results.json", "r+") as f:
        data = json.load(f)
        data["Snippets"].append(result_dict)
        f.seek(0)
        json.dump(data, f)
        f.truncate()
        f.close()


    data = json.load(open("results.json", "r"))
    
    template_path = 'IndividualTemplate.docx'  # path to your template word document
    output_path = 'snippet_output.docx'  # path to save the updated word document

    update_word_doc(template_path, data, output_path)


def final_summary(path, embeddings):
    # Initialize list to store documents
    docs = []


    # Load the text from results.json
    loader = TextLoader("results.json", encoding="utf-8")

    # Split the text into documents
    docs.extend(loader.load_and_split())


    # Chunk the files
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    
    # Store the embeddings in a vector store
    db = FAISS.from_documents(texts, embeddings)

    # Retrieve the embeddings from the vector store
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    # Load the model
    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k") 



    # Load the template that is prompting the agent
    template='''
            You are a code peer reviewer and a software security reviewer. 
            I want you to write me a summary of what the code does and whether there are any considerable
            security concerns within the code that would cause harm to my computer or the network my computer is on.
            It is crucial that your responses include all fields below, leaving none missing. The format of your response should be a python dictionary like the following:

            {
            "Summary": <Short summary of what the code does>
            "Security_Concerns": <Whether or not there are security concerns and if there are what are they>
            "Security_Rating": <A security rating from the following: [Very Low, Low, Moderate, High, Very High]>
            "Justification": <Why you gave it the security rating chosen>
            }

            Remember, it's absolutely critical to include all fields in your response and in the dictionary format.
            '''

    # Create a Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever
    )

    # Create a list to store the chat history 
    chat_history = []

    # Generate a question and response from the model
    result = qa({"question": template, "chat_history": chat_history})


    result_dict = json.loads(result['answer'])
    result_dict["Name"] = "Final Summary"

    
    
    # Append the result to the results.json file
    with open("results.json", "r+") as f:
        data = json.load(f)
        data["Snippets"].append(result_dict)
        f.seek(0)
        json.dump(data, f)
        f.truncate()
        f.close()


    data = json.load(open("results.json", "r"))

    data["Snippets"].sort(key=lambda x: x["Name"] != "Final Summary")
    
    template_path = 'IndividualTemplate.docx'  # path to your template word document
    # last folder in the path if a path or the file name without the extension if in the same folder
    output_path = extract_name(path) + '_SUMMARY.docx'  # path to save the updated word document

    update_word_doc(template_path, data, output_path)

def main():

    # Load .env file
    load_dotenv(find_dotenv())
    st.title("Cyber Digital Revolution: Code Security Reviewer")

    # Create empty results.json
    with open("results.json", "w") as f:
        json.dump({"Snippets": []}, f)
        f.close()


    # Load all repository files. Here we assume this notebook is downloaded as the part of the langchain fork and we work with the python files of the langchain repo.
    path = st.text_input('Enter the path to your code folder or file here:')

    # root_dir = "python-docx-template-master/docxtpl"

    embeddings = OpenAIEmbeddings()

    # Check if folder path is valid
    if path:
        if os.path.isdir(path):
            # This is a directory, process all files in it

            # Get the total number of files in the directory and its subdirectories
            total_files = sum([len(files) for r, d, files in os.walk(path)])

            # os.walk yields a tuple (dirpath, dirnames, filenames) for each directory it visits
            files_processed = 0
            for dirpath, dirnames, filenames in os.walk(path):
                
                for index, file in enumerate(filenames, start=1):
                    process_file(dirpath, file, embeddings)
                    st.write(f"Processed {file} [{index}/{total_files}]")
                    files_processed += 1

            st.write("Finished processing all files!")
        elif os.path.isfile(path):
            # This is a single file, process only this file
            dirpath, filename = os.path.split(path)
            process_file(dirpath, filename, embeddings)
            st.write(f"Processed {filename}")
        else:
            st.write("The path does not exist. Please enter a valid directory or file path.")
    else:
        st.write("Please enter a path to your code folder or file.")

    st.write("Generating final summary...")

    final_summary(path, embeddings)

    st.write("Finished generating final summary!")

    # Delete snippet_output.docx
    os.remove("snippet_output.docx")

    
if __name__ == "__main__":
    main()