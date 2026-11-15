
import os
import faiss

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LLM and Embeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

# Prompt
from langchain_core.prompts import ChatPromptTemplate

# Output parser
from langchain_core.output_parsers import PydanticOutputParser

# Memory and Chain
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_classic.chains import LLMChain

# Vector Store
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


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
    VectorStoreRetrieverMemory ##stores the conversation
)

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings ##vectorizes the data
)

from langchain_community.vectorstores import FAISS ##similarity search
# 2. Initialize embeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

init_kb=["""Traveling is not just about visiting new destinations; its about embracing new cultures, experiencing diverse cuisines ,and opening your mind to the world's endless adventures. packing efficiently is key to stress-free travel. Always check the weather forecast for your destination and pack clothes accordingly .For Paris,it's common for rain showers to according to the weather.
If you're visiting Paris,some must-see attractions include the eiffel tower which is open late into the evening and even sparkles on the hour at night the louvre museum , home to the french cuisine is the world-famous , and in Paris you'll find delights such as french bagguets ,buttery croissants ,escargot (snails), coq au vin,duck confit ,and of course ,an array of pastries.For transportation ,Paris has an excellent pulic transit system.from charles de gaulle airport ,you can reach the city center by a train,airport shuttle ,taxi . Learning a few french phrases -like "bonjour" (hello),"Merci"(thank you),and "s'ill vous plait (please),can make your interactions smother and more enjoyable .always travel with your umbrella
"""]


vectorstore=FAISS.from_texts(init_kb,embeddings)

retriever=vectorstore.as_retriever(search_kwargs={"k":3})

memory=VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="history",
    input_key="user_input",
    return_messages=True
)
prompt=ChatPromptTemplate.from_template(
    "you are a helpful travel assistant .\n"
    "Relevent chat history :\n{history}\n"
    "User:{user_input}\n"
    "Respond only in the following format:\n{format_instructions}\n"
    "Do NOT add any explanation-only the response in the specified format."
)

user_messages = [
    "I am going to Paris next week. Any must-see attractions?",
    "Should I bring an umbrella this time of year?",
    "Also, what are local foods I can't miss?",
    "Is the Eiffel Tower open in the evening?",
    "How do I get from the airport to the city center?",
]

for msg in user_messages:
    history_str=memory.load_memory_variables({"user_input":msg})["message"]

    raw_prompt=prompt.format(
        history=history_str,
        user_input=msg,
        format_instructions=reply_parser.get_format_instructions()
    )

    raw_response=llm.invoke(raw_prompt)
    print("User:",msg)
    print("Raw:",raw_response.content)
    print("-"*40)

    memory.save_content(
    {"user_input":msg},
    {"output":raw_response.content}
    )


