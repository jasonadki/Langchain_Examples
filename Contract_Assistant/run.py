from langchain.llms import OpenAI
from langchain.llms import Ollama

from langchain import PromptTemplate
from langchain.chains import LLMChain

import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def evaluate_contract(statement: str) -> tuple:
    # llm = OpenAI(model_name="gpt-4")
    llm = Ollama(base_url='http://172.27.176.190:6969',model="llama2:7b-chat")

    prompt = PromptTemplate(
        input_variables=["contract_clause"],
        template="""
        You are a seasoned lawyer specializing in contract law. You are hyper focused on finding loopholes and ambiguities in contracts to avoid completing word and saving the company money.

        Review the following contractual clause: {contract_clause}

        Analyze it for potential legal vulnerabilities, ambiguities, or loopholes. Provide feedback on how you would modify the clause to make it more airtight.
        If you can not find any loop holes respond with "No loopholes found." and nothing else.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    evaluation = chain.run(contract_clause=statement)
    evaluation = evaluation.replace("\n", "")

    if evaluation == "No loopholes found.":
        return evaluation, ""
    
    feedback, _, suggested_modification = evaluation.partition(". ")
    if not suggested_modification:
        suggested_modification = feedback
        feedback = ""
    return feedback, suggested_modification

def condense_contract(statement: str) -> str:
    llm = OpenAI(model_name="gpt-4")

    prompt = PromptTemplate(
        input_variables=["contract_statement"],
        template="""
        You are a lawyer with years of experience in drafting and revising contracts.

        Given the contract statement: {contract_statement}

        Please provide a shorter and more concise version of the statement, ensuring it retains its legal accuracy and intent.
        All critical details should remain intact in the condensed version.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(contract_statement=statement)
    response = response.replace("\n", "")
    return response


def update_statement(original_statement: str, feedback: str, provided_statement: str) -> str:
    llm = OpenAI(model_name="gpt-4")

    prompt = PromptTemplate(
        input_variables=["original_statement", "feedback", "provided_statement"],
        template="""
        You are a lawyer with years of experience in drafting and revising contracts.

        Given the contract statement: {provided_statement}
        And the provided feedback: {feedback}
        I need you to make the statement contractually air tight. 
        Please refer to the original statement: {original_statement} to ensure all critical details are retained.
        Do not add unnecessary, redundant, or irrelevant information and ensure to keep the original intent of the statement.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    revised_statement = chain.run(original_statement=original_statement, feedback=feedback, provided_statement=provided_statement)
    return revised_statement.replace("\n", "")

def make_air_tight(original_statement: str, feedback: str, provided_statement: str, current_iteration: str) -> str:
    if feedback == "No loopholes found.":
        return current_iteration

    MAX_ITERATIONS = 2
    iteration_count = 0

    while feedback != "No loopholes found." and iteration_count < MAX_ITERATIONS:
        revised_statement = update_statement(original_statement, feedback, provided_statement)
        feedback, _ = evaluate_contract(revised_statement)
        
        st.write(f"Iteration {iteration_count + 1}: Revised Statement: {revised_statement}")
        
        iteration_count += 1
        current_iteration = revised_statement
        provided_statement = revised_statement

    if iteration_count == MAX_ITERATIONS and feedback != "No loopholes found.":
        st.write("Reached maximum iterations. The contract might still have loopholes.")
    
    return current_iteration


        

def main():
    st.title("Cyber Digital Revoluion:")
    st.title("Contract Assistant")

    choice = st.radio("Choose an action:", ["Condense", "Evaluate"])
    statement = st.text_area("Enter the contractual statement:")

    if 'evaluated_result' not in st.session_state:
        st.session_state.evaluated_result = ""

    if st.button("Submit"):
        if choice == "Evaluate":
            feedback, suggested_modification = evaluate_contract(statement)
            st.session_state.evaluated_result = feedback
            st.subheader("Feedback:")
            st.write(feedback)
            st.subheader("Suggested Modification:")
            st.write(suggested_modification)
        else:
            result = condense_contract(statement)
            st.subheader("Result:")
            st.write(result)

    if st.session_state.evaluated_result and choice == "Evaluate":
        if st.button("Make it air tight"):
            feedback, suggested_modification = evaluate_contract(statement)
            air_tight_result = make_air_tight(statement, feedback, statement, suggested_modification)
            st.subheader("Air Tight Result:")
            st.write(air_tight_result)


if __name__ == "__main__":
    main()
