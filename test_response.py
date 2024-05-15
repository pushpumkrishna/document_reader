import os
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if __name__ == "__main__":
    print("RAG!")
    index_name = "llamaindex-pilot"
    pinecone_index = Pinecone().Index(
        index_name=index_name,
        host="https://llamaindex-pilot-zzdpsaq.svc.aped-4627-b74a.pinecone.io",
    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler()
    callback_manager = CallbackManager(handlers=[llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )

    query = "How to install from source"
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print("Response:: ", response)
