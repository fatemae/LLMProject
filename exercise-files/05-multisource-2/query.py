from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.vectorstores import Chroma
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)
llm_function_calling = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Prompt with Query analysis
class Search(BaseModel):
    """Search for information about a clothing category."""

    query: str = Field(
        ...,
        description="Query to look up",
    )
    category: str = Field(
        ...,
        description="Category to look things up for. Should be `SHOES` or `SHIRTS`.",
    )

system = """
You have the ability to issue search queries to get information to help answer user information. 
if you answer general inquiries, refer to the company name only without the clothes category (shirts, shoes ...).
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
structured_llm = llm_function_calling.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm 

# Prompt for Retrieval and Generation tasks
template: str = """/
    You are a customer support specialist /
    question: {question}. 
    You assist users with general inquiries based on {context} /
    """

# define prompt
system_message_prompt_template = SystemMessagePromptTemplate.from_template(template)
human_message_prompt_template = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"], 
    template="{question}"
)
chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_prompt_template, 
    human_message_prompt_template
])


# Create Index & Connect to multiple data sources
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

raw_text_shoes = TextLoader('./exercise-files/05-multisource-2/docs/faq_shoes.txt').load()
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=100,
    chunk_overlap=0
)
documents = text_splitter.split_documents(raw_text_shoes)
vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings, 
    collection_name="shoes"
)
retriever_shoes = vectorstore.as_retriever(search_kwargs={"k":1})

raw_text_shirts = TextLoader('./exercise-files/05-multisource-2/docs/faq_shirts.txt').load()
documents = text_splitter.split_documents(raw_text_shirts)
vectorstore_shirts = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="shirts"
)
retriever_shirts = vectorstore_shirts.as_retriever(search_kwargs={"k":1})

# Retrieval with Query analysis
retrievers = {
    "SHOES": retriever_shoes,
    "SHIRTS": retriever_shirts,
}

def select_retriever_query_analysis(question):
    """Select a retriever based on the query analysis."""
    structured_output = query_analyzer.invoke(question)
    retriever = retrievers.get(structured_output.category)
    return retriever

def query(user_query: str):
    """Final chain to query, retrieve information and generate augmented response."""
    retriever = select_retriever_query_analysis(user_query)
    return (
        {
            "context": retriever,
            "question": lambda x: x
        }
        | chat_prompt_template
        | llm
        | StrOutputParser()
    ).invoke(user_query)

response = query("how long do we have to return shirts?")
print(response)