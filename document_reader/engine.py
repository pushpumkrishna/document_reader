import os

import pinecone
from dotenv import load_dotenv
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
from pinecone import Pinecone, ServerlessSpec
import nltk

# nltk.download('averaged_perceptron_tagger')

load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENV"),
)
# pc.create_index(
#     name="llamaindex-documentation-helper",
#     dimension=1536,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="GCP", region="us-central1"),
# )
# pinecone_index = pc.Index("llamaindex-documentation-helper")


if __name__ == "__main__":
    print("Going o injest Pinecone documentation")
    UnstructuredReader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="./", file_extractor={"./html_pages-tmp": UnstructuredReader()}
    )
    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=100, chunk_overlap=20)
    node = node_parser.get_nodes_from_documents(documents=documents)

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )
    index_name = "llamaindex-documentation-helper"
    pinecone_index = pinecone.Index(
        index_name=index_name,
        api_key=os.environ.get("PINECONE_API_KEY"),
        host=os.environ.get("PINECONE_HOST"),
    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    print("Finished Ingesting Documents")

    pass
