
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)


# ==========================================
# 2. STRUCTURED OUTPUT
# ==========================================

class AssistantReply(BaseModel):

    reply: str = Field(
        description="The assistant's helpful and friendly response"
    )


# ==========================================
# 3. INITIALIZE CHECKPOINTER
# ==========================================

# InMemorySaver stores conversation state in application memory.
# It saves messages and retrieves them using a thread_id.

checkpointer = InMemorySaver()


# ==========================================
# 4. CREATE THE AGENT
# ==========================================

agent = create_agent(

    # The LLM responsible for generating responses
    model=llm,

    # Instructions defining the assistant's behavior
    system_prompt=(
        "You are a helpful travel assistant. "
        "Remember previous conversations and keep "
        "your responses personal and consistent."
    ),

    # Defines the expected response structure.
    # LangChain handles structured output automatically.
    response_format=AssistantReply,

    # Enables short-term conversation memory
    checkpointer=checkpointer
)


# ==========================================
# 5. CONVERSATION
# ==========================================

# Every conversation needs an identifier.
# Messages using the same thread_id share the same history.

config = {
    "configurable": {
        "thread_id": "travel_conversation_1"
    }
}

user_messages = [
    "I am going to Paris next week. Any must-see attractions?",
    "Should I bring an umbrella this time of year?",
    "Also, what are local foods I can't miss?"
]

for msg in user_messages:

    # Invoke the agent with the current message.
    # The checkpointer automatically retrieves
    # the previous messages for this thread.

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": msg}
            ]
        },
        config=config
    )

    # Retrieve the structured Pydantic response
    reply = result["structured_response"]

    print("User:", msg)
    print("AI:", reply.reply)
    print("-" * 40)
