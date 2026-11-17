import sqlite3

conn =sqlite3.connect("sample.db")
cursor=conn.cursor()
cursor.execute("DROP TABLE IF EXISTS employees;")
cursor.execute("""
CREATE TABLE employees(
id INTEGER PRIMARY KEY,
name TEXT,
department TEXT,
salary INTEGER)""")

cursor.execute("INSERT INTO employees (name,department ,salary) VALUES ('Alice','Engineering','9000')")
cursor.execute("INSERT INTO employees (name,department ,salary) VALUES ('Bob','Marketing','7000')")
cursor.execute("INSERT INTO employees (name,department ,salary) VALUES ('Charlie','HR','6000')")
cursor.execute("INSERT INTO employees (name,department ,salary) VALUES ('Diana','Engineering','9500')")

conn.commit()
conn.close()

import os
from dotenv import load_dotenv
# from pydantic import BaseModel, Field #output model with some fields
from langchain_google_genai import ChatGoogleGenerativeAI #to chat with the model
from langchain_core.prompts import ChatPromptTemplate #to write our prompt template
from langchain_community.utilities import SQLDatabase

# Load environment variables from the .env file
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5 #you need creativity
)

db= SQLDatabase.from_uri("sqlite:///sample.db")

sql_prompt= ChatPromptTemplate.from_template(
    "You are an expert sql developer"
    "translate the following user question into a sql query for sqlite "
    "only output valid sql(no markdown,no code fences,no explanations)"
    "table schema :employees(id,name,department,salary)"
    "User question:{user_question}\n"
    "SQL:"
)

sql_chain=sql_prompt|llm

user_question="Who earns the highest salary in the Engineering Departement?"

response = sql_chain.invoke({
    "user_question": user_question
})

raw_sql = response.content.strip()

print("Generated sql:",raw_sql)

def clean_sql(sql:str)-> str:
    return(
        sql.replace("''''sql","")
        .replace("'''","")
        .strip()
    )

sql_query=clean_sql(raw_sql)
print("clean sql",sql_query)

with sqlite3.connect("sample.db") as conn:
    cursor = conn.cursor()
    cursor.execute(sql_query)
    resulted_row=cursor.fetchall()

print("sql result:",resulted_row)