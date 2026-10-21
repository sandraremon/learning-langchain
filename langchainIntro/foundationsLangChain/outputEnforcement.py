
import os

from pydantic import BaseModel, Field #output model with some fields
from langchain_google_genai import ChatGoogleGenerativeAI #to chat with the model
from langchain_core.prompts import ChatPromptTemplate #to write our prompt template
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
model="gemini-3.8-flash",
google_api_key=os.getenv("GOOGLE_API_KEY")
)
#to make sure output is specific even if there is a problem with the llm
#schema 
#propmt -> llm -> output parser
class weatherReport(BaseModel):
    city:str=Field(description="city name")
    temperature_c:float=Field(description="expected temperature in c")
    advice:str=Field(description="what to wear / carry?")

#we will link llm output with pydantic class
parser = PydanticOutputParser(pydantic_object=weatherReport)

safe_parser= OutputFixingParser.from_llm(parser=parser,llm=llm)

prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "Give me the current weather for {city} in JSON.\n"
    )
])

chain=prompt|llm|safe_parser

print(chain.invoke({"city":"london"}))