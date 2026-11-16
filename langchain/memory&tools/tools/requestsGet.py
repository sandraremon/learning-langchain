import os
from dotenv import load_dotenv
# from pydantic import BaseModel, Field #output model with some fields
from langchain_google_genai import ChatGoogleGenerativeAI #to chat with the model
from langchain_core.prompts import ChatPromptTemplate #to write our prompt template
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.tools.requests.tool import RequestsGetTool
# Load environment variables from the .env file
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3#you need creativity
)

prompt=ChatPromptTemplate.from_template(
    "the user will provide a person's name."
    "Return ONLY the Nationalize.io API url to PREDICT their nationality"
    "BASE URL: https://api.nationalize.io/?name=NAME\n"
    "User :{user_input}\n"
    "API URL:"
)
chain=prompt|llm
user_name="Mohamed"
api_url=chain.invoke({"user_input":user_name})['text'].strip()
print("Generated API URL:",api_url)

#tools
requests_wrapper=RequestsWrapper()
get_tool=RequestsGetTool(
    requests_wrapper=requests_wrapper,
    allow_dangerous_requests=True
)

result=get_tool.run(api_url) #call the get tool
print(f"\nPredicted nationality for'{user_name}':\n",result)


