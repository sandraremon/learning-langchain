import os
from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import true
load_dotenv()
#choosing the model to use for the LLM
llm = ChatGoogleGenerativeAI(
model="gemini-3.8-flash",
google_api_key=os.getenv("GOOGLE_API_KEY")
)
#memory that summarizes the conversation
memory=ConversationBufferMemory()

chatbot=ConversationChain(llm=llm,memory=memory,verbose=True) #verbose to monitor the transaction occuring in chat

print(chatbot.predict(input="will i want to practice spanish."))
print(chatbot.predict(input="Ask me a question about food."))
print(chatbot.predict(input="Correct my answer:'Yo gusto la pizza.'"))