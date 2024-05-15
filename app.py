import os
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone
import streamlit as st


load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

llama_debug = LlamaDebugHandler()
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    print("RAG!")
    index_name = "llamaindex-pilot"
    pinecone_index = Pinecone().Index(
        index_name=index_name,
        host="https://llamaindex-pilot-zzdpsaq.svc.aped-4627-b74a.pinecone.io",
    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )

    return index


index = get_index()

if "chat_engine" not in st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5,
        threshold_cutoff=0.7,
    )
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, verbose=True, node_postprocessor=[postprocessor]
    )

st.set_page_config(
    page_title="Chat with Llamaindex Docs, powered by llama-index",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with Llamaindex Docs 💬🦙, powered by Pushpum Krishna")

if "message" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "Assistant", "content": "Ask me a question about India"}
    ]
if prompt := st.chat_input("Your question here !!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "Assistant":
    with st.chat_message("Assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            nodes = [node for node in response.source_nodes]
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source node{i + 1}: score= {node.score}")
                    st.write(node.text)
            st.write(response.response)
            message = {"role": "Assistant", "content": response.response}
            st.session_state.messages.append(message)
