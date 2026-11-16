import os
from dotenv import load_dotenv
# from pydantic import BaseModel, Field #output model with some fields
from langchain_google_genai import ChatGoogleGenerativeAI #to chat with the model
from langchain_core.prompts import ChatPromptTemplate #to write our prompt template
from langchain_experimental.utilities import SerpAPIWrapper

# Load environment variables from the .env file
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5 #you need creativity
)

serp_api=SerpAPIWrapper(serpapi_api_key="placeholder")

prompt=ChatPromptTemplate.from_template(
    "You are a helpful assistant.Rewrite the user's question as a single,concise google search query"
    "Do NOT answer the question do NOT sat anything except the search query itself"
    "User:{user_input}"
    "Search Query:"
)
chain=prompt|llm

user_question="what is the time now in egypt?"

search_query=chain.invoke({"user_input":user_question})['text'].strip

if not search_query:
    search_query=user_question
print("Generated search query:",search_query)

serp_result=serp_tool.run(search_query)

print("search results:",serp_result)