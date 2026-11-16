#types of chains:sequential,routing,parallel,retrieval qa chain(in rag pipelines"chat with knowledge base) 

import os

from pydantic import BaseModel, Field #output model with some fields
from langchain_google_genai import ChatGoogleGenerativeAI #to chat with the model
from langchain_core.prompts import ChatPromptTemplate #to write our prompt template
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.chains import SequentialChain, LLMChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
model="gemini-3.8-flash",
google_api_key=os.getenv("GOOGLE_API_KEY"),
temperature=0.7
)

class citySuggestion(BaseModel):
    city:str

class Outline(BaseModel):
    day1:str=Field(description="Description of activities for day 1")
    day2:str=Field(description="Description of activities for day 2")
    day3:str=Field(description="Description of activities for day 3")

city_parser=PydanticOutputParser(pydantic_object=citySuggestion)
output_parser=PydanticOutputParser(pydantic_object=Outline)
 #by .partial we enforce format rules

city_prompt=ChatPromptTemplate.from_messages([
    ("human","Recommend a city fir a traveler seeking:{travel_goal}.\n\n{format_instructions}")
    ]).partial(format_instructions=city_parser.get_format_instructions())

city_chain=LLMChain(llm=llm,prompt=city_prompt,output_key="city")


outline_prompt=ChatPromptTemplate.from_messages([("human","Give a short 3 day outline for visiting {city} for this travel goal.\n\n{format_instructions}")]).partial(format_instructions=output_parser.get_format_instructions)

output_chain=LLMChain(llm=llm,prompt=outline_prompt,output_key="outline")

chain=SequentialChain(
    chains=[city_chain,output_chain],
    input_variables=["travel_goal"],
    output_variables=["city","outline"],
    verbose=True
)
result=chain.invoke({"travel_goal":"adventure travel"})
print(result)