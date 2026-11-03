from src.data.chunker import chunk_documents
from src.data.loader import load_documents
from src.retrieval.vector_store import create_collection, get_client, index_chunks


def main() -> None:
    chunks = chunk_documents(load_documents())
    print(f"{len(chunks)} chunks")
    client = get_client()
    create_collection(client, recreate=True)
    index_chunks(client, chunks)
    print("indexed", client.count("handbook").count, "points")


if __name__ == "__main__":
    main()
