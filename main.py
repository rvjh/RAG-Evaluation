from core.rag import RAGPipeline
from evaluation.evaluation import evaluate_full


def run_demo():
    print("\n🚀 Initializing RAG pipeline...\n")

    rag = RAGPipeline()
    rag.initialize()

    print("\n💬 Running test query...\n")

    query = "What are neural networks?"
    answer, docs = rag.run(query)

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== RETRIEVED CONTEXT ===\n")
    for i, d in enumerate(docs):
        print(f"[Chunk {i+1}] {d.page_content}\n")

    print("\n📊 Running evaluation...\n")

    results = evaluate_full(rag)

    print("\n=== RETRIEVAL METRICS ===")
    print(results["retrieval"])

    print("\n=== GENERATION METRICS (RAGAS) ===")
    print(results["generation"])


if __name__ == "__main__":
    run_demo()
	
	