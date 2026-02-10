from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

load_dotenv()

model = ChatOpenAI()

loader = WebBaseLoader("https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

# pull rag prompt
prompt = hub.pull("rlm/rag-prompt")

# Construct a chain to answer questions on your data
web_search_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | model
    | StrOutputParser()
)
