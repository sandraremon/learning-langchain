docs=[
    "Pyramids of Giza: Ancient Egyptian monumental tombs built over 4,500 years ago for the pharaohs."
"Eiffel Tower: An iconic iron tower in Paris, France, built in 1889."
"Amazon: One of the world's largest technology and e-commerce companies, known for online shopping and cloud computing."
]


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

# Load environment variables from the .env file
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3 #you need creativity
)



embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

vectore_store=FAISS.from_texts(docs,embeddings)

retriever=vectore_store.as_retriever(search_kwargs={"k":2}) #set the no of queries back 2 paragraphs similar

user_question="When were the Pyramids of Giza built?"

relevant_docs=retriever._get_relevant_documents(user_question)

context="\n\n".join(doc.page_content for doc in relevant_docs)

print("Retrieved context for assistant:\n",context)

q_a_prompt=ChatPromptTemplate.from_template(
"Answer the user questions using only the provided context."
"If the answer isnt present,say so.\n\n"
"Context:{context}" #top 2 chunks retrieved from vector store
"User question:{user_question}"
"Answer:"
)

qa_chain=q_a_prompt|llm

response=qa_chain.invoke({"context":context,
                          "user_question":user_question})

print("\nFinal Answer:",response.content.strip())






