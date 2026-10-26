
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda


# Load environment variables from .env
load_dotenv()


# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


# Schema for travel classification
class RouterInput(BaseModel):
    travel_type: Literal["fun", "medical"] = Field(
        ...,
        description="Type of travel: fun or medical"
    )


# Schema for fun travel plans
class FunTravelPlan(BaseModel):
    city: str = Field(..., description="Recommended city")
    day1: str = Field(..., description="Main activity for day 1")
    day2: str = Field(..., description="Main activity for day 2")
    day3: str = Field(..., description="Main activity for day 3")


# Schema for medical travel plans
class MedicalTravelPlan(BaseModel):
    city: str = Field(..., description="Recommended city")
    hospital: str = Field(..., description="Suitable hospital or clinic")
    day1: str = Field(..., description="Medical appointments for day 1")
    day2: str = Field(..., description="Recovery plan for day 2")
    day3: str = Field(..., description="Optional relaxation for day 3")


# Initialize output parsers
router_input_parser = JsonOutputParser(pydantic_object=RouterInput)
fun_parser = JsonOutputParser(pydantic_object=FunTravelPlan)
medical_parser = JsonOutputParser(pydantic_object=MedicalTravelPlan)


# Prompt templates
router_prompt_b = ChatPromptTemplate.from_template(
    """A user wants to travel.
    Classify their request as either "fun" or "medical".

    Travel goal: {travel_goal}

    Follow these format instructions:
    {format_instructions}"""
)

fun_prompt_b = ChatPromptTemplate.from_template(
    """Create a 3-day fun travel plan for this goal:
    {travel_goal}

    Recommend a suitable city and activities for each day.

    Follow these format instructions:
    {format_instructions}"""
)

medical_prompt_b = ChatPromptTemplate.from_template(
    """Create a 3-day medical travel plan for this goal:
    {travel_goal}

    Recommend a suitable city and hospital.
    Include medical appointments and recovery considerations.
    Do not invent verified medical credentials or guarantee treatment suitability.

    Follow these format instructions:
    {format_instructions}"""
)


# Build the classification chain
router_prompt = router_prompt_b.partial(
    format_instructions=router_input_parser.get_format_instructions()
)

router_chain = router_prompt | llm | router_input_parser


# Handler for fun travel
def handle_fun(travel_goal: str) -> dict:
    fun_prompt = fun_prompt_b.partial(
        format_instructions=fun_parser.get_format_instructions()
    )

    fun_chain = fun_prompt | llm | fun_parser

    return fun_chain.invoke({"travel_goal": travel_goal})


# Handler for medical travel
def handle_medical(travel_goal: str) -> dict:
    medical_prompt = medical_prompt_b.partial(
        format_instructions=medical_parser.get_format_instructions()
    )

    medical_chain = medical_prompt | llm | medical_parser

    return medical_chain.invoke({"travel_goal": travel_goal})


# Route the request to the correct handler
def route(output: dict) -> dict:
    travel_type = output["travel_type"]
    travel_goal = output["travel_goal"]

    if travel_type == "fun":
        return handle_fun(travel_goal)

    elif travel_type == "medical":
        return handle_medical(travel_goal)

    else:
        return {
            "error": "Sorry, I can only handle fun or medical travel plans."
        }


# Build the complete application pipeline
router_app = {
    "travel_type": router_chain | RunnableLambda(
        lambda result: result["travel_type"]
    ),
    "travel_goal": lambda x: x["travel_goal"]
} | RunnableLambda(route)


# Execute the application
if __name__ == "__main__":
    result = router_app.invoke(
        {"travel_goal": "I want to go for a fun vacation"}
    )

    print(result)
