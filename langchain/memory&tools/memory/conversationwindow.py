import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Prompt components
from langchain_core.prompts import (
    ChatPromptTemplate,
  
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

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

memory=ConversationBufferWindowMemory(
    memory_key="chat_history",
    input_key="user_input",
    k=2, #only 2 exchanges are remembered
    return_messages=True
    )

prompt=ChatPromptTemplate.from_template(
        "You are a helpful travel assistant. "
        "You Remember only the lase few exchanges of our conversation to keep it relevant"
        "Chat history so far:\n{chat_history}\n"
        "User input: {user_input}\n"
        "Respond ONLY using this format:\n{format_instructions}"
        "Dont add any explanation only the responses specified in the specified format"
)

chain=LLMChain(
    llm=llm,
    prompt=prompt.partial(format_instructions=reply_parser.get_format_instructions()),
    memory=memory,
    output_parser=reply_parser
)

user_messages = [
    "I am going to Paris next week. Any must-see attractions?",
    "Should I bring an umbrella this time of year?",
    "Also, what are local foods I can't miss?"
    "Is the eiffel tower open in the evening ?"
    "how do i get from the airport to the city center?"
    "could you call my first message?"
]

for msg in user_messages:
    raw_response = chain.llm.invoke(chain.prompt.format_(chat_history=memory.buffer,user_input=msg))
    print("User:",msg)
    print("RAW:",raw_response.content)
    print("-"*40)
    