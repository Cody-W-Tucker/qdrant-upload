"""
title: Langchain RAG
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: langchain, langchain_core, langchain_openai, langchain_qdrant
"""

import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(...)
        OPENAI_MODEL: str = Field(default="gpt-4o-mini")
        QDRANT_URL: str = Field(default="http://localhost:6333")

    def __init__(self):
        self.name = "Journal RAG"
        self.vector_store = None
        self.llm = None
        self.valves = self.Valves(**{
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "QDRANT_URL": os.getenv("QDRANT_URL", "http://qdrant.homehub.tv"),
        })

    async def on_startup(self):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        
        # Connect to the Qdrant vector store
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name="obsidian_docs",
            url=self.valves.QDRANT_URL,  # Use the URL from the valves attribute
        )
        self.llm = ChatOpenAI(model=self.valves.OPENAI_MODEL)
        
    async def on_shutdown(self):
        # Optional cleanup code when the server shuts down.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.chains import (
            create_history_aware_retriever,
            create_retrieval_chain,
            create_stuff_documents_chain,
        )

        # Perform a similarity search
        retriever = self.vector_store.as_retriever()

        # Create a history-aware retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Create the RAG chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "\\n\\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create document combining chain using stuff method
        question_answering_chain = create_stuff_documents_chain(
            llm=self.llm, 
            prompt=qa_prompt
        )

        # Create final RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answering_chain
        )

        response = rag_chain.invoke({
            "chat_history": messages,
            "input": user_message
        })

        return response["answer"]