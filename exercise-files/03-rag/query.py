import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

template: str = """/
    You are a customer support specialist /
    question: {question}. 
    You assist users with general inquiries based on {context} /
    and  technical issues. /
    """

# define prompt
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["user_query","context"], 
    template="{question}")
prompt_template = ChatPromptTemplate.from_messages(
    [
       system_message_prompt,
        human_message_prompt,
    ]
)

# init model
model = ChatOpenAI(model="gpt-4o-mini")

# indexing
def load_split_documents():
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    raw_text = TextLoader("./exercise-files/03-rag/docs/faq.txt").load()
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=30,
        chunk_overlap=0,
        length_function=len,
    )
    chunks =  text_splitter.split_documents(raw_text)
    return chunks

# convert to embeddings
def load_embeddings(documents, user_query):
    """Create a vector store from a set of documents."""
    embedding = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embedding)
    docs = db.similarity_search(user_query, k=3)
    print("Retrieved docs:", docs)
    return db.as_retriever()


def generate_response(retriever, query):
    """Generate a response to a user query."""
    chain =( {
        "context": retriever,
        "question": lambda x: x
    } | prompt_template | model | StrOutputParser())
    return chain.invoke(query)


def query(query):
    """Query the model with a user query."""
    documents=load_split_documents()
    retriever = load_embeddings(documents, query)
    response = generate_response(retriever, query)
    return response
    
