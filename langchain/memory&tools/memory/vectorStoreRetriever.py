
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


# ==========================================
# 3. EMBEDDING MODEL
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Get the number of dimensions dynamically
dimension = len(
    embeddings.embed_query("dimension check")
)


# ==========================================
# 4. VECTOR STORE
# ==========================================

# Create an EMPTY vector database.
# Conversations will be added automatically.

vectorstore = FAISS(

    embedding_function=embeddings,

    index=faiss.IndexFlatL2(dimension),

    docstore=InMemoryDocstore(),

    index_to_docstore_id={}
)


# ==========================================
# 5. RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(

    search_type="similarity",

    # Retrieve the 2 most relevant documents
    search_kwargs={"k": 2}
)


# ==========================================
# 6. CONVERSATION MEMORY
# ==========================================

memory = VectorStoreRetrieverMemory(

    retriever=retriever,

    memory_key="chat_history",

    input_key="user_input",

    # Return retrieved memories as a string
    return_docs=False
)


# ==========================================
# 7. PROMPT TEMPLATE
# ==========================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        "You are a helpful travel assistant.\n"
        "Use the relevant previous conversations "
        "to answer the user's question.\n\n"

        "Relevant conversation history:\n"
        "{chat_history}\n\n"

        "Respond using this format:\n"
        "{format_instructions}"
    ),

    ("human", "{user_input}")

])

prompt = prompt.partial(
    format_instructions=reply_parser.get_format_instructions()
)


# ==========================================
# 8. CREATE THE CHAIN
# ==========================================

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    output_key="text"
)


# ==========================================
# 9. CONVERSATION
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

    # Parse the raw JSON response
    parsed = reply_parser.parse(
        response["text"]
    )

    print("User:", msg)

    print("RAW:", response["text"])

    print("AI:", parsed.reply)

    print("-" * 40)
