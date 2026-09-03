import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="google/gemma-4-31b-it:free",
    api_key=os.getenv("OPEN_ROUTER_KEY"),
    base_url = "https://openrouter.ai/api/v1",
    temperature=0.2
)

response = llm.invoke("Say exactly: OpenRouter works!")

print(response.content)

