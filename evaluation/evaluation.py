import json
import numpy as np

from config.config import Config
from utils.utils import timed
from langchain_groq import ChatGroq


# -------------------------
# Load dataset
# -------------------------
def load_golden():
    with open(Config.GOLDEN_DATASET, "r") as f:
        return json.load(f)


# -------------------------
# Retrieval metrics
# -------------------------
def precision_at_k(docs, gt, k=3):
    docs = docs[:k]
    hits = sum(gt.lower() in d.lower() for d in docs)
    return hits / k


def recall_at_k(docs, gt, k=3):
    docs = docs[:k]
    return 1.0 if any(gt.lower() in d.lower() for d in docs) else 0.0


def hit_rate(docs, gt, k=3):
    docs = docs[:k]
    return 1.0 if any(gt.lower() in d.lower() for d in docs) else 0.0


def mrr(docs, gt, k=3):
    docs = docs[:k]
    for i, d in enumerate(docs):
        if gt.lower() in d.lower():
            return 1 / (i + 1)
    return 0.0


# -------------------------
# LLM judge (Groq)
# -------------------------
llm = ChatGroq(
    model=Config.LLM_MODEL,
    api_key=Config.GROQ_API_KEY
)


def llm_score(prompt):
    """returns 0-1 score from LLM"""
    res = llm.invoke(prompt).content
    try:
        return float(res.strip())
    except:
        return 0.5


# -------------------------
# Faithfulness (Groq judge)
# -------------------------
def faithfulness(answer, contexts):
    prompt = f"""
You are an evaluator.

Context:
{contexts}

Answer:
{answer}

Rate faithfulness (0 to 1 only):
- 1 = fully supported by context
- 0 = hallucinated

Return ONLY number.
"""
    return llm_score(prompt)


# -------------------------
# Relevancy (Groq judge)
# -------------------------
def relevancy(question, answer):
    prompt = f"""
Question: {question}
Answer: {answer}

Rate relevance of answer to question (0 to 1 only).

Return ONLY number.
"""
    return llm_score(prompt)


# -------------------------
# Retrieval evaluation
# -------------------------
@timed
def evaluate_retrieval(rag):

    data = load_golden()

    precision_scores = []
    recall_scores = []
    hit_scores = []
    mrr_scores = []

    for item in data:

        docs = rag.retrieve(item["question"])
        texts = [d.page_content for d in docs]

        gt = item["ground_truth"]

        precision_scores.append(precision_at_k(texts, gt))
        recall_scores.append(recall_at_k(texts, gt))
        hit_scores.append(hit_rate(texts, gt))
        mrr_scores.append(mrr(texts, gt))

    return {
        "Precision@K": round(float(np.mean(precision_scores)), 2),
        "Recall@K": round(float(np.mean(recall_scores)), 2),
        "Hit Rate": round(float(np.mean(hit_scores)), 2),
        "MRR": round(float(np.mean(mrr_scores)), 2)
    }


# -------------------------
# Generation evaluation
# -------------------------
@timed
def evaluate_generation(rag):

    data = load_golden()

    faith_scores = []
    rel_scores = []

    for item in data:

        answer, docs = rag.run(item["question"])

        context_text = " ".join([d.page_content for d in docs])

        faith_scores.append(
            faithfulness(answer, context_text)
        )

        rel_scores.append(
            relevancy(item["question"], answer)
        )

    return {
        "Faithfulness": round(float(np.mean(faith_scores)), 2),
        "Relevancy": round(float(np.mean(rel_scores)), 2)
    }


# -------------------------
# Full evaluation
# -------------------------
@timed
def evaluate_full(rag):

    return {
        "retrieval": evaluate_retrieval(rag),
        "generation": evaluate_generation(rag)
    }