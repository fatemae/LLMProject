from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from retrievers.vector_search_index import rag_chain
from retrievers.web_search import web_search_chain

load_dotenv()

# load the LLM
llm = ChatOpenAI()  

# define the tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool 
def vector_search_query(query):
    """search queries using vector search"""
    return rag_chain.invoke(query)

@tool
def web_search_query(query):
    """search queries using web search for additional information"""
    return web_search_chain.invoke(query)

tools = [get_word_length, vector_search_query, web_search_query]

# # create the prompt

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events, If you don't know the answer, just say I don\'t know. Do not try to make up an answer"
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# # bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# create the agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# # run the agent

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(query):
    question = "How to secure a MongoDB Atlas Cluster"
    answer = agent_executor.invoke({"input": query})
    return answer['output']

# answer = run_agent("How to create a vector store")
# print(answer)