import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

app = FastAPI()

llm = ChatOpenAI(
    base_url="https://agent-tracing-demo-resource.services.ai.azure.com/api/projects/agent-tracing-demo/openai/v1",
    api_key=os.environ.get("MICROSOFT_FOUNDRY"),
    model="gpt-5-mini",
    temperature=0,
)

#Tool 1
search = DuckDuckGoSearchRun()


agent = create_agent(
    model=llm,
    tools=[search]
)

class Query(BaseModel):
    message: str    


@app.get("/")
def root():
    return {"status": "Agent is running"}

@app.post("/invoke")
def invoke(query: Query):
    result = agent.invoke({
        "messages": [("user", query.message)]
    })

    return {
        "response": result["messages"][-1].content
    }