
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


# 1. Define the output models
class PackagingList(BaseModel):
    items: list[str] = Field(
        description="Packaging items for the trip"
    )
    note: str = Field(
        description="Extra tip"
    )


class LocalFoods(BaseModel):
    foods: list[str] = Field(
        description="Must-try foods in this city"
    )
    note: str = Field(
        description="Food tip"
    )


# 2. Create output parsers
packaging_parser = PydanticOutputParser(
    pydantic_object=PackagingList
)

food_parser = PydanticOutputParser(
    pydantic_object=LocalFoods
)


# 3. Create prompts
packaging_prompt = ChatPromptTemplate.from_template(
    "You are a helpful travel packing assistant.\n"
    "Suggest a packing list for a 1-week trip to {city}.\n"
    "{format_instructions}"
).partial(
    format_instructions=packaging_parser.get_format_instructions()
)

food_prompt = ChatPromptTemplate.from_template(
    "You are a helpful local food guide.\n"
    "Suggest must-try foods for a 1-week trip to {city}.\n"
    "{format_instructions}"
).partial(
    format_instructions=food_parser.get_format_instructions()
)


# 4. Create modern LangChain chains
packaging_chain = packaging_prompt | llm | packaging_parser

food_chain = food_prompt | llm | food_parser


# 5. Run both chains in parallel
parallel = RunnableParallel({
    "packaging_list": packaging_chain,
    "local_foods": food_chain
})


# 6. Invoke the parallel chain
city = "Tokyo"

result = parallel.invoke({"city": city})

print(result)

# Optional: Print each result separately
print("\nPACKAGING LIST")
print(result["packaging_list"])

print("\nLOCAL FOODS")
print(result["local_foods"])
