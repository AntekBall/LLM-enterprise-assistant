from qdrant_client import QdrantClient, models

from src.data.chunker import Chunk
from src.retrieval.embedder import embed_documents, embed_query

COLLECTION = "handbook"
VECTOR_SIZE = 1024 


def get_client(url: str = "http://localhost:6333") -> QdrantClient:
    return QdrantClient(url=url)


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
        client.create_payload_index(
            COLLECTION, "department", models.PayloadSchemaType.KEYWORD
        )


def index_chunks(client: QdrantClient, chunks: list[Chunk], batch_size: int = 256) -> None:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embed_documents([c.text for c in batch])
        points = [
            models.PointStruct(
                id=i + j,
                vector=vector,
                payload={
                    "text": c.text,
                    "source": c.source,
                    "department": c.department,
                    "title": c.title,
                    "section": c.section,
                    "chunk_id": c.chunk_id,
                },
            )
            for j, (c, vector) in enumerate(zip(batch, vectors))
        ]
        client.upsert(COLLECTION, points=points)


def search(
    client: QdrantClient,
    query: str,
    departments: list[str] | None = None,
    top_k: int = 5,
) -> list[models.ScoredPoint]:
    query_filter = None
    if departments is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="department", match=models.MatchAny(any=departments)
                )
            ]
        )
    return client.query_points(
        COLLECTION,
        query=embed_query(query),
        query_filter=query_filter,
        limit=top_k,
    ).points