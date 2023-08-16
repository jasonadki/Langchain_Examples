from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
import json
import os

app = Flask(__name__)
CORS(app)
load_dotenv(find_dotenv())


def create_db_from_json(jsonFile: str) -> FAISS: 
    if jsonFile is not None:

        # load the json UploadedFile(id=1, name='Color_Coding_1_decomposed.json', type='application/json', size=15514)
        with open(jsonFile, 'r') as file:
            jsonFile = json.load(file)

        

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

    print(f'Query: {query}')

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="gpt-4")

    prompt = PromptTemplate(
    input_variables=["statement", "docs"],
    template="""
    You are a helpful assistant that can determine the appropriate color coding for given statements. 
    Determine the correct color code for the following statement: {statement} 
    Using the specified color coding guide: {docs}

    It is crucial that your responses include all fields below, leaving none missing. The format of your response should be like the following:

    - "color": The chosen color code.
    - "reasoning": Explanation based on the docs.

    Remember, it's absolutely critical to include all fields in your response and that it be given in a JSON format.
    """,

    )



    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(statement=query, docs=docs_page_content)
    response = response.replace("\n", "")
    print(f'Response: {response}')
    print(f'docs: {docs}')
    return response, docs

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    json_content = data.get("pdf")
    query = data.get("query")
    if not (json_content and query):
        return jsonify({"error": "pdf or query missing"}), 400
    

    db = create_db_from_json(json_content)
    response, docs = get_response_from_query(db, query)
    
    # Extracting details for each document for front-end display
    doc_details = [{
        "description": doc.metadata['Description'],
        "code": doc.metadata['Code'],
        "document_name": doc.metadata['Document_Name']
    } for doc in docs]

    return jsonify({
        "response": response,
        "documents": doc_details
    })


if __name__ == '__main__':
    app.run(port=5000)
