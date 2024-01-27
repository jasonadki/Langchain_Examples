# from langchain_community.chat_models import ChatOpenAI
import copy
import json
from langchain_openai import ChatOpenAI
# from langchain.llms import Ollama

from flask import Flask, render_template, request, jsonify

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = ChatOpenAI(model_name="gpt-4")


def condense_contract(statement: str) -> str:

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

    response = chain.invoke(input=statement)
    response = response['text'].replace("\n", "")
    return response







def evaluate_contract(statement: str) -> tuple:
    
    prompt = PromptTemplate(
        input_variables=["contract_clause"],
        template="""
        You are a seasoned lawyer specializing in contract law. You are hyper focused on finding loopholes and ambiguities in contracts to avoid completing work and to save the company money.

        Review the following contractual clause: {contract_clause}

        Analyze it for potential legal ambiguities, or loopholes. Provide feedback on how you would modify the clause to make it more airtight.
        If you cannot find any loopholes, respond with "No loopholes found." and nothing else.
    
        It is very important that the format you respond in is a string dictionary with the following information:
        "feedback": <your observations and lawyer opinions on the contractual clause>
        "suggested_modification": <A replacement clause that you would use to replace the original clause>

        Your response will be used in a program so the format is very important that it is followed exactly as shown above in key value dictionary.
        """,
    )


    chain = LLMChain(llm=llm, prompt=prompt)

    evaluation = chain.invoke(input={"contract_clause": statement})

    feedbackResponse = evaluation['text']
    
    # Convert the string response to a dictionary
    feedback_dict = json.loads(feedbackResponse)


    feedback = feedback_dict.get('feedback', '')
    suggested_modification = feedback_dict.get('suggested_modification', '')

    return feedback, suggested_modification






def update_statement(original_statement: str, current_feedback: str, suggested_statement: str) -> str:

    prompt = PromptTemplate(
        input_variables=["original_statement", "feedback", "provided_statement"],
        template="""
        You are a lawyer with years of experience in drafting and revising contracts.

        Given the contract statement: {suggested_statement}
        And the provided feedback: {current_feedback}
        I need you to make the statement contractually air tight. 
        Please refer to the original statement: {original_statement} to ensure all critical details are retained.
        Do not add unnecessary, redundant, or irrelevant information and ensure to keep the original intent of the statement.
        All I want back is the revised statement, I do not need any feedback or commentary on the revised statement.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    revised_statement = chain.invoke(input={"original_statement": original_statement, "current_feedback": current_feedback, "suggested_statement": suggested_statement})
    
    return revised_statement['text']




def make_air_tight(original_statement: str, current_feedback: str, suggested_statement: str, max_iterations: int) -> tuple:
    MAX_ITERATIONS = int(max_iterations)
    iteration_count = 0
    iteration_details = []


    while current_feedback != "No loopholes found." and iteration_count < MAX_ITERATIONS:
        revised_statement = update_statement(original_statement, current_feedback, suggested_statement)
        current_feedback, suggested_modification = evaluate_contract(revised_statement)
        
        iteration_details.append(f"Iteration {iteration_count + 1}: Revised Statement: {revised_statement}")
        
        iteration_count += 1
        provided_statement = copy.deepcopy(revised_statement)

    if iteration_count == MAX_ITERATIONS and current_feedback != "No loopholes found.":
        iteration_details.append("Reached maximum iterations. The contract might still have loopholes.")


    return provided_statement, iteration_details





app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render your HTML file


@app.route('/condense', methods=['POST'])
def condense():
    statement = request.json['statement']
    result = condense_contract(statement)
    return jsonify({'result': result})


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json

    feedback = evaluate_contract(data['statement'])

    return jsonify({'feedback': feedback[0], 'suggested_modification': feedback[1]})


@app.route('/make-airtight', methods=['POST'])
def airtight():
    data = request.json
    result, iteration_details = make_air_tight(
        original_statement = data['original_statement'],
        current_feedback = data['current_feedback'],
        suggested_statement = data['suggested_statement'],
        max_iterations = data['max_iterations']
        )
    return jsonify({'result': result, 'iteration_details': iteration_details})


        


if __name__ == "__main__":
    app.run(debug=False)
