#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes
from langchain.schema.runnable import RunnableLambda
from langserve.schema import CustomUserType
from langchain.schema import (
    AIMessage,
)
from qa_retrieval import retrieval_chain
from dotenv import load_dotenv
import json
import openai
import os

load_dotenv()

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

class Response(CustomUserType):
    content: dict
    callback_events: list
    metadata: dict


def parseResponse(response: AIMessage) -> str:
    """Sample function that expects a Response type which is a pydantic model"""
    return  response.content

# create server
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces"
)

# create runnable
chain = prompt | model | RunnableLambda(parseResponse)

# create routes

add_routes(app, model, path="/openai")

add_routes(app, chain, path="/joke")

add_routes(app, retrieval_chain, path="/qa")


# run server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)