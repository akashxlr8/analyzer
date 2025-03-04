from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL


import os, json
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

class CodeOutput(BaseModel):
    code: str = Field(description="Valid Python code to execute")
    explanation: Optional[str] = Field(description="Explanation of the code")

class CodeEnabledLLM:
    def __init__(self):
        # Initialize Azure OpenAI client
        self.llm = AzureChatOpenAI(
            azure_deployment="bfsi-genai-demo-gpt-4o",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Initialize Python REPL tool
        self.python_repl = PythonREPL()

        # Define the tool
        self.tools = [
            Tool(
                name="python_repl",
                func=self.python_repl.run,
                description="Useful for executing python code. Use this to perform data analysis, calculations, and any other tasks that require code execution. Input should be a valid python code snippet."
            )
        ]

    def analyze_question(self, df: pd.DataFrame, question: str, category: str) -> str:
        """Analyze a question by generating and executing Python code."""
        try:
            # Get a glimpse of the dataset (head)
            df_head = df.head().to_string()

            # Construct the prompt for the LLM
            prompt = f"""
            You are an AI data analyst. You have access to a python_repl tool to execute python code.
            You are working with a Pandas DataFrame. Here's a glimpse of the data (DataFrame.head()):
            {df_head}

            Your task is to answer the following question: {question}
            Category: {category}

            To accomplish this, you should:
            1.  Write Python code that uses the Pandas library to analyze the data and answer the question.
            2.  Provide a brief explanation of the code.
            3.  Return the code and explanation in a structured JSON format.

            Make sure to include all necessary imports (e.g., pandas).
            Wrap your code in ```python blocks.

            {{
                "code": "your python code here",
                "explanation": "brief explanation of the code"
            }}
            """

            # Invoke the LLM with the prompt
            response = self.llm.invoke(prompt)

            # Check if the response is a list and convert to string
            if isinstance(response.content, list):
                response_content = "".join(str(item) for item in response.content)
            else:
                response_content = str(response.content)

            try:
                # Parse the JSON response using Pydantic
                code_output = CodeOutput.model_validate_json(response_content.encode('utf-8'))
                code = code_output.code
                explanation = code_output.explanation

                # Execute the code using the python_repl tool
                if code:
                    try:
                        result = self.python_repl.run(code)
                        return f"Explanation: {explanation}\nResult: {result}"
                    except Exception as e:
                        return f"Error executing code: {str(e)}"
                else:
                    return "No code was generated."

            except Exception as e:
                return f"Error parsing JSON response: {str(e)}\nRaw response: {response_content}"

        except Exception as e:
            return f"Error during analysis: {str(e)}"

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Create a sample DataFrame
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [6, 7, 8, 9, 10]}
    df = pd.DataFrame(data)

    # Initialize the CodeEnabledLLM
    code_llm = CodeEnabledLLM()

    # Example question
    question = "What is the average of col1 multiplied by the sum of col2?"
    category = "Mathematical Calculation"

    # Analyze the question
    analysis_result = code_llm.analyze_question(df, question, category)
    print("Analysis Result:", analysis_result)
