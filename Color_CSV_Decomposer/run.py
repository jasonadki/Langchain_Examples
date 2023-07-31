from uuid import uuid4
import ast
import os
import json
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

from dotenv import load_dotenv, find_dotenv
import streamlit as st


def process_file(dirpath, file):

    results = []

    # Get the path to the csv file
    dirpath = os.path.join(dirpath, file)

    # Read in the csv file and print the rows
    df = pd.read_csv(dirpath)
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Load the model
    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    
    # Load the template that is prompting the agent
    with open('snippet_prompt_template.txt', 'r') as f:
        template= f.read()

    # Iterate through the rows of the csv file
    for index, row in df.iterrows():
        # Create dict from the row
        result_dict = row.to_dict()

        messages = [
            SystemMessage(content = template),
            HumanMessage(content = result_dict["Remarks"])
        ]

        response = model(messages)

        response_list = ast.literal_eval(response.content)

        for i in range(len(response_list)):
            results.append({
                "UUID": str(uuid4()),
                "Description": response_list[i][0],
                "Code": response_list[i][1],
            })

    # Get file name without extension
    file_name_no_ext = os.path.splitext(file)[0]

    # Save results to a json file
    with open(f"{file_name_no_ext}_decomposed.json", "w") as f:
        json.dump({"Information": results}, f, indent=4)


def main():

    # Load .env file
    load_dotenv(find_dotenv())
    
    st.title("Cyber Digital Revolution: Color Code Decomposer")

    results = []

    path = st.text_input('Enter the path to your code folder or file here:')

    # Check if folder path is valid
    if path:
        if os.path.isfile(path):
            # This is a single file, process only this file
            dirpath, filename = os.path.split(path)
            process_file(dirpath, filename)
            st.write(f"Processed {filename}")
        else:
            st.write("The path does not exist. Please enter a valid directory or file path.")
    else:
        st.write("Please enter a path to your code folder or file.")


if __name__ == "__main__":
    main()
