# src/search.py
from typing import List, Dict, Any


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store, embedding_manager):
        """
        Initialise the retriever

        Args:
            vector_store: VectorStore instance (Chroma-backed)
            embedding_manager: EmbeddingManager instance
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        print(f"Retrieving documents for query: '{query}'")

        # 1️⃣ Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            # 2️⃣ Query vector store
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs: List[Dict[str, Any]] = []

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            if not documents:
                print("No documents found")
                return []

            # 3️⃣ Process results
            for rank, (doc_id, document, meta, dist) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1
            ):
                similarity_score = 1 - dist  # cosine similarity

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": meta,
                        "similarity_score": similarity_score,
                        "distance": dist,
                        "rank": rank
                    })

            print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []