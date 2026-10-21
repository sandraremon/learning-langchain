import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
#choosing the model to use for the LLM
llm = ChatGoogleGenerativeAI(
model="gemini-3.8-flash",
google_api_key=os.getenv("GOOGLE_API_KEY")
)
#customizing input
#initializing the prompt template
prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helpful travel agent"),
    #no search this is from knowledge base no real time
    ("user","i want to visit {destination} for {days} days, suggest me least offers")
])

chain=prompt | llm

destination=input("enter preferered destination: ")
days=input("enter number of days: ")

response=chain.invoke({"destination":destination,"days":days})
# If response.content is a list of dicts:
if isinstance(response.content, list):
    text = "".join(
        part["text"] for part in response.content if isinstance(part, dict) and "text" in part
    )
    print(text)
else:
    print(response.content)

#templates ->leaving placeholders for user to fill the values