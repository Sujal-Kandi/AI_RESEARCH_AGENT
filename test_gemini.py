import os 
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=os.getenv("GEMINI_KEY"),
    temperature=0.2
)

response = llm.invoke("Say exactly: Gemini works!")

print(response.content)

