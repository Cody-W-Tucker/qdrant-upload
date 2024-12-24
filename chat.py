import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Initialize the OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Connect to the Qdrant vector store
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="obsidian_docs",
    url="http://qdrant.homehub.tv",  # Remote Qdrant server URL
)

llm = ChatOpenAI(model="gpt-4o-mini")

message = """
Using provided context and questions, synthesize a first-person response from the author. The context consists of semantically split snippets from their journal, reflecting the author's deepest thoughts. The answers should be crafted to implicitly reflect Piaget's focus on cognitive development and adaptation, Nietzsche's emphasis on personal growth and self-realization, and Jordan Peterson's narrative-focused approach from "Maps of Meaning" without explicitly mentioning these frameworks. The synthesized response should be comprehensive, cohesive, and suitable for a living document intended to provide personalized assistance.
{question}

Rules: Don't mention people's names or sensitive information. Generalize the context to ensure privacy.

Context:
{context}
"""

# Perform a similarity search
query = "Write a blog post that explains what it’s like being misunderstood and how the great thinkers were often misunderstood"

# Retrieve the relevant documents
retriever = vector_store.as_retriever()

# Format the documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke(query)

print(response)