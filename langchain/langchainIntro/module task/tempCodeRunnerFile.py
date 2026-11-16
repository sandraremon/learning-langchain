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
# ------------------------------------------------------------------------
# 1. Pydantic Models (Strict Data Contracts)
# ------------------------------------------------------------------------

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




# ------------------------------------------------------------------------
# 2. Mock Data Retrieval Functions
# ------------------------------------------------------------------------

def fetch_structured_financials(query: str) -> str:
    """Mock document loader simulating highly structured financial database queries."""
    return f"Financial Data DB Record: Q3 EPS $1.25, Net Revenue $45B. Associated query: '{query}'."

def fetch_unstructured_news(query: str) -> str:
    """Mock document loader simulating unstructured market news wire extraction."""
    return f"News Wire Extract: Analysts upgrade target stock amidst strong AI integration rally. Associated query: '{query}'."

# ------------------------------------------------------------------------
# 3. Parallel Data Aggregation
# ------------------------------------------------------------------------

def build_parallel_data_loader() -> RunnableParallel:
    """
    TODO: Construct a concurrent execution pipeline using RunnableParallel.
    This pipeline must aggregate outputs from `fetch_structured_financials` 
    and `fetch_unstructured_news` simultaneously to prevent slow, rigid extended chains.
    """

    return RunnableParallel({
        # "query": RunnableLambda(lambda inputs: inputs["query"]),
        # "intent": RunnableLambda(lambda inputs: inputs["intent"]),
        "financial_summary": RunnableLambda(
            lambda inputs: fetch_structured_financials(inputs["query"])
        ),
        "market_news": RunnableLambda(
            lambda inputs: fetch_unstructured_news(inputs["query"])
        ),
    })
    # Hint: Return a RunnableParallel instance mapping keys to the mock functions above.






# ------------------------------------------------------------------------
# 5. Pipeline Assembly
# ------------------------------------------------------------------------

def build_fincorp_pipeline():
    """
    TODO: Assemble the final LCEL orchestration pipeline.
    
    Requirements:
    1. Instantiate a ChatOpenAI model using 'gpt-4o-mini' to keep inference fast and lightweight.
    2. Create dynamic, reusable inference chains using PromptTemplate with explicit placeholders 
       where the user input and retrieved data will be injected.
    3. Utilize the RunnableParallel data aggregator built above.
    4. Wrap your routing logic in a RunnableLambda.
    5. Link the user input, data aggregator, router/prompts, model, and Pydantic output parser 
       into a single executable workflow using LCEL syntax (e.g., using the | operator).
    """
    inputs=inputs
    
    # 1. Initialize the LLM (gpt-4o-mini)
    llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
    
    # 2. Initialize PydanticOutputParsers for both models
    financial_parser = PydanticOutputParser(
    pydantic_object=FinancialSummary
)
    news_parser = PydanticOutputParser(
            pydantic_object=MarketNews
    )

    # 3. Define PromptTemplates for financial summaries and market news
    financial_prompt = PromptTemplate(
       " You are a financial analysis assistant .\n"

       " Analyze the user's query using the financial data provided\n"

       " User query:{query}\n"
        "{format_instructions}\n"
        
).partial(
        format_instructions=financial_parser.get_format_instructions()
    )
    news_prompt = PromptTemplate(
       " You are a  market news aggregator .\n"

       " Analyze the user's query using the market new data provided\n"

       " User query:{query}\n"
        "{format_instructions}\n"
        
).partial(
        format_instructions=news_parser.get_format_instructions()
    )
     # 4. Assemble the LCEL chain
    financial_chain=financial_prompt|llm|financial_parser
    news_chain=news_prompt|llm|news_parser
    # ------------------------------------------------------------------------
# 4. Conditional Routing
# ------------------------------------------------------------------------
    
    def route_query(inputs: Dict[str, Any]):
        """
        TODO: Implement a conditional routing function to create router chains.
        Evaluate the classified user query and dynamically navigate workflow branches
        (e.g., executing a financial inference chain vs. a news inference chain).
        """
        inputs=inputs["intent"]
        if inputs == "financial" :
            return financial_chain
        elif inputs == "news" :
            return news_chain
        else:
            "sorry invalid choice"
        # Hint: Use the inputs dictionary to determine the classification intent.
        # Return the appropriate LangChain sequence (Prompt | LLM | Parser) based on the intent.
    route=RunnableLambda(route_query)
    pipeline = (
    RunnablePassthrough.assign(
        data=build_parallel_data_loader
    )
    | RunnableLambda(
        lambda inputs: {
            "query": inputs["query"],
            "intent": inputs["intent"],
            **inputs["data"]
        }
    )
    | route
    )
    

    
   

    

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment variables before execution
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    print("Starting NexusFlow AI - FinCorp RAG Architecture PoC...")
    
    # Example execution (Uncomment to test once implemented):
    pipeline = build_fincorp_pipeline()
  
    # 
    test_input = {
        "query": "Impact of the new generative AI orchestration on Q4 profit margins",
        "intent": "financial" # This field should be utilized by your router logic
    }
    print(build_parallel_data_loader().invoke(test_input))

    
    try:
        result = pipeline.invoke(test_input)
        print("Pipeline Execution Successful!")
        print(result)
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")