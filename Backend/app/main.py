import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun          # + added WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper                        # + added
from langchain_core.tools import tool      
import numexpr                                                                       # + added

app = FastAPI()

openrouter_key = os.environ.get("OPENROUTER_API_KEY")

if not openrouter_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set!")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    temperature=0,
)
# Tool 1: web search
search = DuckDuckGoSearchRun()

# Tool 2: encyclopedia lookups — good for facts, definitions, historical info
# + added
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '2 * (3 + 4)' or '15% of 200'."""
    try:
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Couldn't evaluate that: {e}"

agent = create_agent(
    model=llm,
    tools=[search,wikipedia, calculator]         
)

class Query(BaseModel):
    message: str    


@app.get("/")
def root():
    return {"status": "Agent is running"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
def invocations(query: Query):
    return invoke(query)


@app.post("/invoke")
def invoke(query: Query):
    result = agent.invoke({
        "messages": [("user", query.message)]
    })

    return {
        "response": result["messages"][-1].content
    }
