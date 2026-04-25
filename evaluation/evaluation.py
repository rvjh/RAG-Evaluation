import json
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from config.config import Config
from utils.utils import timed


def load_golden():
    with open(Config.GOLDEN_DATASET, "r") as f:
        return json.load(f)


def precision_at_k(docs, gt, k=3):
    docs = docs[:k]
    return sum(gt in d for d in docs) / k


def recall_at_k(docs, gt, k=3):
    docs = docs[:k]
    return 1.0 if any(gt in d for d in docs) else 0.0


def mrr(docs, gt, k=3):
    docs = docs[:k]
    for i, d in enumerate(docs):
        if gt in d:
            return 1 / (i + 1)
    return 0.0


@timed
def evaluate_retrieval(rag):

    data = load_golden()

    p, r, m = [], [], []

    for item in data:
        docs = rag.retrieve(item["question"])
        texts = [d.page_content for d in docs]

        gt = item["ground_truth"]

        p.append(precision_at_k(texts, gt))
        r.append(recall_at_k(texts, gt))
        m.append(mrr(texts, gt))

    return {
        "Precision@K": np.mean(p),
        "Recall@K": np.mean(r),
        "MRR": np.mean(m)
    }


@timed
def evaluate_generation(rag):

    data = load_golden()

    qs, ans, ctx, gt = [], [], [], []

    for item in data:

        a, docs = rag.run(item["question"])

        qs.append(item["question"])
        ans.append(a)
        ctx.append([d.page_content for d in docs])
        gt.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": qs,
        "answer": ans,
        "contexts": ctx,
        "ground_truth": gt
    })

    return evaluate(dataset, metrics=[faithfulness, answer_relevancy])


@timed
def evaluate_full(rag):
    return {
        "retrieval": evaluate_retrieval(rag),
        "generation": evaluate_generation(rag)
    }