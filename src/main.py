from src.data_loader import DataLoader
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.search import RAGRetriever

def main():
    loader = DataLoader()
    docs = loader.load_documents()
    chunks = loader.split_documents(docs)

    texts = [doc.page_content for doc in chunks]

    embed_manager = EmbeddingManager()
    embeddings = embed_manager.generate_embeddings(texts)

    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vector_store, embed_manager)

    results = retriever.retrieve(
        query="What is LangChain?",
        top_k=3,
        score_threshold=0.2
    )

    for r in results:
        print(f"\nRank {r['rank']} | Score: {r['similarity_score']:.3f}")
        print(r["content"][:300])

if __name__ == "__main__":
    main()