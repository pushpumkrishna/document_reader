from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader


def web_reader(url: str) -> str:
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()
    response = str(query_engine.query("What is your name?"))

    return response

    pass


if __name__ == "__main__":
    print("Process started !!!")

    load_dotenv()
    page_url = "https://magazine.atavist.com/watch-it-burn-france-europe-carbon-fraud-scam-vat-betrayal/?src=longreads"
    result = web_reader(page_url)
