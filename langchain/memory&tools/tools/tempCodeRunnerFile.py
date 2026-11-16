import os
from dotenv import load_dotenv
# from pydantic import BaseModel, Field #output model with some fields
from langchain_google_genai import ChatGoogleGenerativeAI #to chat with the model
from langchain_core.prompts import ChatPromptTemplate #to write our prompt template
from langchain_core.output_parsers import PydanticOutputParser
from langchain_experimental.utilities import PythonREPL

# Load environment variables from the .env file
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3 #you need creativity
)

prompt=ChatPromptTemplate.from_template(
    "You are a python coding assistant .Given the users request,reply only with the valid python code to solve the problem"
    "Dont add any explanation ,comments or markdown"
    "always assign the answer to a variable if needed ,and  Output the final answer using print(...) "
    "Just the code-no text befor or after it.\n"
    "User:{user_input}"
)

chain=prompt|llm

#run as any python interpreter

python_repl=PythonREPL()

user_message="calculate the average of 3,7,10 and 15"
reponse=chain.invoke({"user_input":user_message})

print("LLM output:\n", reponse.content)

def extract_python_code(text):
    lines=text.strip.split('\n')
    if lines[0].startwith("'''"):
        return "\n".join(lines[1:-1])
    return text.strip()

code_to_run=extract_python_code(reponse['text'])
print(code_to_run)

result=python_repl.run(code_to_run)
print("Execution result:",result)