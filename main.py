from core.rag import RAGPipeline
from evaluation.evaluation import evaluate_full


rag = RAGPipeline()
rag.initialize()

answer, docs = rag.run("What is encryption?")
print("\nANSWER:\n", answer)

results = evaluate_full(rag)

print("\n=== RETRIEVAL ===")
print(results["retrieval"])

print("\n=== GENERATION ===")
print(results["generation"])