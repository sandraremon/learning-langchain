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

# Chain
from langchain_classic.chains import LLMChain


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


# ==========================================
# 2. STRUCTURED OUTPUT
# ==========================================

class AssistantReply(BaseModel):

    reply: str = Field(
        description="The assistant's helpful response"
    )


reply_parser = PydanticOutputParser(
    pydantic_object=AssistantReply
)


from langchain_classic.memory import (
    ConversationSummaryMemory
)


# ==========================================
# 3. CONVERSATION MEMORY
# ==========================================

memory = ConversationSummaryMemory(

    # LLM used to generate summaries
    llm=llm,

    memory_key="chat_history",

    input_key="user_input",

    # output_key="text",

    return_messages=True
)


from langchain_classic.memory import ConversationBufferMemory



# ==========================================
# 4. PROMPT TEMPLATE
# ==========================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        "You are a helpful travel assistant. "
        "Use the previous conversation to answer.\n"
        "Respond using this format:\n"
        "{format_instructions}"
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    ("human", "{user_input}")

])

prompt = prompt.partial(
    format_instructions=reply_parser.get_format_instructions()
)


# ==========================================
# 5. CREATE THE CHAIN
# ==========================================

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    output_key="text"
)


    


# ==========================================
# 6. CONVERSATION
# ==========================================

user_messages = [
    "I am going to Paris next week. Any must-see attractions?",
    "Should I bring an umbrella this time of year?",
    "Also, what are local foods I can't miss?",
    "Is the Eiffel Tower open in the evening?",
    "How do I get from the airport to the city center?",
    "Could you recall my first message?"
]

for msg in user_messages:

    # Execute the complete chain
    response = chain.invoke({
        "user_input": msg
    })

    # Extract the raw LLM response
    raw_response = response["text"]

    # Convert JSON into AssistantReply
    parsed_response = reply_parser.parse(raw_response)

    print("User:", msg)

    print("RAW:", raw_response)

    print("PARSED:", parsed_response.reply)

    # Inspect the updated conversation summary
    print("SUMMARY:", memory.buffer)

    print("-" * 40)
