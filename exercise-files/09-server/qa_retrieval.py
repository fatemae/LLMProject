from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from dotenv import load_dotenv
import os, pprint

load_dotenv()

model = ChatOpenAI()
ATLAS_CONNECTION_STRING = os.getenv("ATLAS_CONNECTION_STRING")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

# initialize MongoDB python client
client = MongoClient(ATLAS_CONNECTION_STRING)

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]


# Load the PDF
loader = PyPDFLoader("https://investors.mongodb.com/node/13176/pdf")
data = loader.load()

# Split PDF into documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(data)

# Print the first document
# print(docs[0].page_content)

# Create the vector store for test.books
vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(disallowed_special=()),
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

# Create the vector index for test.books
query = "How to enroll in a course?"
results = vector_search.similarity_search(query)

# Instantiate Atlas Vector Search as a retriever
qa_retriever = vector_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1, "score_threshold": 0.75}
)

# Define a basic question-answering prompt template
prompt_template = """

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)


def format_result(output):
    return output["result"]


# Create the question-answering model
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

retrieval_chain = (
    {"context": qa_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

