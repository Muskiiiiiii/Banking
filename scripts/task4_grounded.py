"""Task 4 on its own: grounded generation with the calibrated "I don't know" fallback.
Run calibration first (python -m scripts.calibrate_threshold), then:
    python -m scripts.task4_grounded
"""
import config
from rag.index import ensure_indexes
from rag.retrieval import GroundedGenerator

# The same in-scope queries are reused for Task 5 (precision/recall), with their
# hand-labelled relevant documents.
EVAL_QUERIES = {
    "What is the minimum monthly income required for a personal loan?": {"01_loan_eligibility"},
    "Do I pay a foreclosure charge if I prepay my personal loan?": {"08_prepayment_penalty"},
    "Which documents are needed to complete KYC?": {"04_kyc_documents"},
    "What should I do if I see an unauthorised transaction on my card?": {"05_fraud_dispute"},
    "How is my EMI calculated and when is the first EMI due?": {"02_emi_calculation"},
    "What credit score gets the lowest interest rate, and what hurts my credit score?":
        {"07_interest_rate_slabs", "10_credit_score_factors"},
}
OUT_OF_SCOPE_QUERY = "What is the best pizza place in Bengaluru?"


def main():
    ensure_indexes()
    print(f"MOCK_LLM={config.MOCK_LLM}  collection={config.DEPLOYED_COLLECTION}  "
          f"top_k={config.TOP_K}  IDK_THRESHOLD={config.IDK_THRESHOLD}")
    gen = GroundedGenerator(use_cache=False)
    for q in list(EVAL_QUERIES) + [OUT_OF_SCOPE_QUERY]:
        r = gen.answer(q)
        tag = "GROUNDED" if r["grounded"] else "FALLBACK (below threshold)"
        print(f"\nQ: {q}\n   top-1 cosine = {r['top1_similarity']:.4f} -> {tag}\n   A: {r['answer']}")


if __name__ == "__main__":
    main()
