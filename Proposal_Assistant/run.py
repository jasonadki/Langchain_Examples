from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv, find_dotenv
import textwrap






load_dotenv(find_dotenv())



def create_db_from_pdf(pdf_path: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    db = FAISS.from_documents(pages, OpenAIEmbeddings())
    return db


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


if __name__ == "__main__":
    # Example usage:
    pdf_url = "Resources/Halcyon - Aeolus.pdf"
    db = create_db_from_pdf(pdf_url)

    query = "How much will the proposal cost?"
    response, docs = get_response_from_query(db, query)
    print(response, docs)