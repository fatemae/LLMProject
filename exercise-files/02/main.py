from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# create a vectorstore & embeddings
vectorstore=FAISS.from_texts(["harrison worked at kensho", "harrison is 50 year old"], embedding=OpenAIEmbeddings())


# querying the vectorstore
query = "Where did harrison work?"
docs = vectorstore.similarity_search(query, top_k=1)
# print("Context:", docs[0].page_content)

# querying as retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":1})
docs = retriever.invoke(query)
# print("Context via retriever:", docs[0].page_content)

retrieval_chain = (
    RunnableMap({
        "context": retriever,
        "question": lambda x: x
    })
    | prompt
    | llm
    | StrOutputParser()
)

result = retrieval_chain.invoke(query)
print(result)