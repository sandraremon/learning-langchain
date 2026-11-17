import os
from dotenv import load_dotenv
# from pydantic import BaseModel, Field #output model with some fields
from langchain_google_genai import ChatGoogleGenerativeAI #to chat with the model
from langchain_core.prompts import ChatPromptTemplate #to write our prompt template
from langchain_community.retrievers import WikipediaRetriever

# Load environment variables from the .env file
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3 #you need creativity
)

user_question="When were the Pyramids of Giza built?"

#an accurate prompt to call wikipedia
search_prompt=ChatPromptTemplate.from_template(
    "Given the user's question,output the wikipedia search phrase.\n"
    "User question:{user_question}\n"
    "Wikipedia search phase:"
)

search_chain=search_prompt|llm

wiki_search_phrase = search_chain.invoke({
    "user_question": user_question
})
print("WikiSearch phrase:",wiki_search_phrase)

wiki_retriever=WikipediaRetriever(top_k_results=3)
docs=wiki_retriever.get_relevant_documents(wiki_search_phrase)

print("\n===Wikipedia Retreived Context ====")
for i,doc in enumerate(docs,1):
    print(f"\nResults {i}:")
    print("Title:",doc.metadata.get('title','N/A'))
    print("Summary:\n",doc.page_content[:400])


context="\n\n".join(doc.page_content for doc in docs)

answer_content=ChatPromptTemplate.from_template(
    "You are an expert research assistant .User the provided wikipedia context to answer the user's question in a clear ,concise paragraph"
    "if the answer is not in the context ,say so .\n\n"
    "wikipedia context:{wikipedia_context}\n\n"
    "user question:{user_question}"
    "Answer:"
)
answer_chain=answer_content|llm


response=answer_chain.invoke(
    {
        "wikipedia_context":context,
        "user_question":user_question
    }
)
print("\n===LLM's wikipedia-Based answer\n")

print(response.content)