from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"],
)

# pc.create_index(
#     name="llamaindex-test",
#     dimension=8,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-west-2"),
# )
# pinecone_index1 = pc.Index("llamaindex-test")

if __name__ == "__main__":
    print("Going to ingest pinecone documentation...")
    UnstructuredReader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(input_dir="../data/html_pages-tmp")

    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )

    pinecone_index1 = "llamaindex-pilot"
    pinecone_index = Pinecone().Index(
        index_name=pinecone_index1,
        host="https://llamaindex-pilot-zzdpsaq.svc.aped-4627-b74a.pinecone.io",
    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    print("finished ingesting...")
