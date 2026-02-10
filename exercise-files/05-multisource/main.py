from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0)


# Prompt with Query analysis
class Search(BaseModel):
    query: str = Field(..., description="Query to look up")
    person: str = Field(
        ..., description="Person to look things up for. Should be `Harrison` or `Ankush`"
    )

system = """You have the ability to issue search queries to get information to help answer users"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human","{question}")
    ]
)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm


# structured_output = query_analyzer.invoke(
#     "Where does Harrison work?")
# print("Structured Output:", structured_output)

# Create Index & Connect to datasource
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = ["Harrison worked at Kensho.", "Harrison is 50 years old."]
vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings, collection_name="harrison")
retriever_harrison = vectorstore.as_retriever(search_kwargs={"k":1})

texts = ["Ankush is 30 years old.", "Ankush works at OpenAI."]
vectorstore_2 = Chroma.from_texts(texts=texts, embedding=embeddings, collection_name="ankush")
retriever_ankush = vectorstore_2.as_retriever(search_kwargs={"k":1})

# docs = vectorstore.similarity_search("Who works at Kensho?", k=1)
# print("Response:", docs[0].page_content)

# Retrieval with Query analysis

retrievers = {
    "Harrison": retriever_harrison,
    "Ankush": retriever_ankush
}

@chain
def custom_chain(question: str) -> str:
    structured_output = query_analyzer.invoke(question)
    retriever = retrievers.get(structured_output.person)
    docs = retriever.invoke(structured_output.query)
    return docs

result = custom_chain.invoke("Where does Ankush work?")
print("Final Result:", result)