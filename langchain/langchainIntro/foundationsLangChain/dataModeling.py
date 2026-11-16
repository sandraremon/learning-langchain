
import os

from pydantic import BaseModel, Field #output model with some fields
from langchain_google_genai import ChatGoogleGenerativeAI #to chat with the model
from langchain_core.prompts import ChatPromptTemplate #to write our prompt template
from langchain_core.output_parsers import PydanticOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
model="gemini-3.8-flash",
google_api_key=os.getenv("GOOGLE_API_KEY")
)
#customizing output
class weatherReport(BaseModel):
    city:str=Field(description="city name")
    temperature_c:float=Field(description="expected temperature in c")
    advice:str=Field(description="what to wear / carry?")
#we will link llm output with pydantic class
parser = PydanticOutputParser(pydantic_object=weatherReport)
#initialized node in the chain called parser following the weather report


#build the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "Give me the current weather for {city} in JSON.\n"
        "{format_instructions}"
    )
]).partial(
    format_instructions=parser.get_format_instructions()
)

##the output is driven is parser

#lets create chain
chain=prompt|llm|parser

city=input("enter the city name:")

report:weatherReport =chain.invoke({"city":city})
print(report)
