
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Prompt components
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# Output parser
from langchain_core.output_parsers import PydanticOutputParser

# Legacy LangChain memory and chain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import LLMChain


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

# Load environment variables from the .env file
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# temperature=0 means more consistent, less random responses.


# ==========================================
# 2. STRUCTURED OUTPUT
# ==========================================

# Define the structure we expect from the LLM.
class AssistantReply(BaseModel):

    # The response must contain a field named "reply"
    # and its value must be a string.
    reply: str = Field(
        description="The assistant's helpful and friendly response"
    )


# Create a parser that converts the LLM response
# into an AssistantReply Pydantic object.
reply_parser = PydanticOutputParser(
    pydantic_object=AssistantReply
)


# ==========================================
# 3. CONVERSATION MEMORY
# ==========================================

memory = ConversationBufferMemory(

    # Name of the variable that will contain
    # the retrieved conversation history.
    # Must match the prompt's MessagesPlaceholder.
    memory_key="chat_history",

    # Identifies the current user input.
    # Used when saving the interaction.
    input_key="user_input",

    # Identifies the LLM output to save.
    output_key="text",

    # Return history as message objects:
    # HumanMessage and AIMessage.
    return_messages=True
)


# ==========================================
# 4. PROMPT TEMPLATE
# ==========================================

prompt = ChatPromptTemplate.from_messages([

    # SYSTEM MESSAGE:
    # Defines the assistant's role and behavior.
    (
        "system",
        "You are a helpful travel assistant. "
        "Remember previous conversations and keep your responses "
        "personal and consistent.\n"
        "Respond using this format:\n{format_instructions}"
    ),

    # CONVERSATION HISTORY:
    # Dynamically inserts previous HumanMessage
    # and AIMessage objects retrieved from memory.
    MessagesPlaceholder(variable_name="chat_history"),

    # CURRENT USER MESSAGE:
    # {user_input} will be replaced with the new user input.
    ("human", "{user_input}")

])

# Fill format_instructions once instead of passing it
# manually every time we invoke the prompt.
prompt = prompt.partial(
    format_instructions=reply_parser.get_format_instructions()
)


# ==========================================
# 5. CREATE THE CHAIN
# ==========================================

chain = LLMChain(

    # The language model responsible for generating responses.
    llm=llm,

    # The prompt that organizes instructions,
    # conversation history, and current input.
    prompt=prompt,

    # Automatically retrieves and saves conversation history.
    memory=memory,

    # Converts the raw LLM output into AssistantReply.
    output_parser=reply_parser,

    # Name of the output variable returned by the chain.
    output_key="text"
)


# ==========================================
# 6. CONVERSATION
# ==========================================

user_messages = [
    "I am going to Paris next week. Any must-see attractions?",
    "Should I bring an umbrella this time of year?",
    "Also, what are local foods I can't miss?"
]

for msg in user_messages:

    # Invoke the COMPLETE chain.
    # We only pass the new user input.
    # Memory automatically supplies the previous history.
    response = chain.invoke({
        "user_input": msg
    })

    print("User:", msg)

    # response["text"] is an AssistantReply object.
    # .reply accesses its string field.
    print("AI:", response["text"].reply)

    print("-" * 40)
