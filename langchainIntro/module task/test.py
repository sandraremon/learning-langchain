import os
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

    # 1. Initialize the LLM (gpt-4o-mini)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

class FinancialSummary(BaseModel):
    """
    TODO: Define strict field definitions and descriptions for financial summaries.
    This acts as a structural contract to remove noise from LLM outputs for downstream 
    reporting tools used by FinCorp.
    """
    company_name: str = Field(...,description="The official ticker or registered name of the entity (e.g., FinCorp Inc., AAPL).")
    quarter:str=Field(...,description="The fiscal quarter and year referenced in the query. Must follow the format  (e.g., Q3 2023).")
    total_revenue :float=Field(description="The total consolidated revenue reported for the specified quarter, represented as a numeric float in millions (e.g., 12450.50).")
    net_income:float=Field(description="The bottom-line net income reported for the specified quarter, represented as a numeric float in millions (e.g., 3120.75).")
    pass

class MarketNews(BaseModel):
    """
    TODO: Define strict field definitions and descriptions for market news.
    This ensures unstructured news is parsed into a reliable, consistent format.
    """
    company_name :str= Field(...,description="The target company being analyzed in the news aggregator.")
    sentiment_score :float=Field(...,description="A calculated sentiment score derived from the aggregated news. Must be a float between -1.0 (Highly Negative) and 1.0 (Highly Positive).")
    top_headlines:list[str]=Field(...,description="A list containing exactly the top 3 most relevant news headlines impacting the company's market perception.")
    pass

def route_query(inputs: Dict[str, Any]):
    """
    TODO: Implement a conditional routing function to create router chains.
    Evaluate the classified user query and dynamically navigate workflow branches
    (e.g., executing a financial inference chain vs. a news inference chain).
    """
    if inputs == "executing a financial inference chain" :
        return financial_chain
    elif inputs == "a news inference chain" :
        return news_chain
    else:
        "sorry invalid choice"
    # Hint: Use the inputs dictionary to determine the classification intent.
    # Return the appropriate LangChain sequence (Prompt | LLM | Parser) based on the intent.
    pass


financial_parser = PydanticOutputParser(
    pydantic_object=FinancialSummary
)
news_parser = PydanticOutputParser(
        pydantic_object=MarketNews
)

financial_prompt = PromptTemplate.from_template(
    "You are a financial analysis assistant.\n User query: {query}\n Financial data: {financial_summary}\n\n"
    "{format_instructions}"
).partial(
    format_instructions=financial_parser.get_format_instructions()
)
news_prompt = PromptTemplate.from_template(
       " You are a  market news aggregator .\n Analyze the user's query using the market new data provided\n""User query:{query}\n"
        "{format_instructions}\n"
        
).partial(
        format_instructions=news_parser.get_format_instructions()
    )

financial_chain=financial_prompt|llm|financial_parser
news_chain=news_prompt|llm|news_parser
test_input = {
     "query": "Impact of the new generative AI orchestration on Q4 profit margins",
        "intent": "financial" # This field should be utilized by your router logic
    }

def route_query(inputs: Dict[str, Any]):
    intent = inputs.get("intent")

    if intent == "financial":
        return financial_chain
    if intent == "news":
        return news_chain

    raise ValueError(f"Expected intent 'financial' or 'news', got {intent!r}")


router_app = RunnableLambda(route_query)

if __name__ == "__main__":
    # Fictional figures for testing the financial branch.
    result = router_app.invoke({
        "intent": "financial",
        "query": "Summarize Example Corp's Q4 2025 financial results.",
        "financial_summary": (
            "Example Corp reported Q4 2025 revenue of 1,250 million "
            "and net income of 180 million."
        ),
    })
    print(result.model_dump())


