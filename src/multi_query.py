"""Multi-query generation and parallel hybrid candidate retrieval.

Generates controlled, targeted query expansions preserving the original query,
and executes parallel multi-query hybrid retrieval across lexical and dense semantic indexes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("campus_pipeline.multi_query")

BRANCH_EXPANSIONS = {
    "CSE": ["Computer Science and Engineering", "CSE", "CS"],
    "ISE": ["Information Science and Engineering", "ISE", "IS"],
    "AIML": ["Artificial Intelligence & Machine Learning", "AIML", "AI ML"],
    "CD": ["Computer Science & Design", "CSD", "CD"],
    "CY": ["Cyber Security", "CY", "CYS"],
    "ECE": ["Electronics and Communication Engineering", "ECE"],
    "EEE": ["Electrical and Electronics Engineering", "EEE"],
    "ME": ["Mechanical Engineering", "ME"],
    "CV": ["Civil Engineering", "CV"],
}


def generate_expanded_queries(
    original_query: str,
    entities: Optional[Dict[str, Any]] = None,
    intent: Optional[str] = None,
    max_queries: int = 4,
) -> List[str]:
    """Generate controlled, high-precision search query variations.

    The original query is ALWAYS retained as the primary query (index 0).
    Produces at most ``max_queries`` variations.
    """
    cleaned = (original_query or "").strip()
    if not cleaned:
        return []

    queries: List[str] = [cleaned]
    seen_normalized: Set[str] = {cleaned.lower()}

    if not entities:
        return queries

    exam = entities.get("exam")
    branch = entities.get("branch")
    quota = entities.get("admission_quota")
    company = entities.get("company")
    data_type = entities.get("data_type")
    counselling_round = entities.get("counselling_round")

    # 1. Intent: Admissions Cutoff
    if intent in {"admissions_cutoff", "admissions"} or data_type in {"previous_cutoff", "cutoff"}:
        exam_term = exam or ("KCET" if "kcet" in cleaned.lower() else "COMEDK" if "comedk" in cleaned.lower() else "")
        branch_term = branch or ""

        if branch_term and exam_term:
            q_var1 = f"RVCE {exam_term} {branch_term} closing rank cutoff"
            q_var2 = f"{exam_term} {branch_term} round cutoff rank RVCE"
            for q_candidate in [q_var1, q_var2]:
                if q_candidate.lower() not in seen_normalized and len(queries) < max_queries:
                    queries.append(q_candidate)
                    seen_normalized.add(q_candidate.lower())

        # Full branch name expansion
        if branch_term in BRANCH_EXPANSIONS:
            full_name = BRANCH_EXPANSIONS[branch_term][0]
            q_var3 = f"RVCE {exam_term} {full_name} cutoff rank".strip()
            if q_var3.lower() not in seen_normalized and len(queries) < max_queries:
                queries.append(q_var3)
                seen_normalized.add(q_var3.lower())

        if counselling_round:
            q_var4 = f"RVCE {exam_term} {branch_term} {counselling_round} cutoff".strip()
            if q_var4.lower() not in seen_normalized and len(queries) < max_queries:
                queries.append(q_var4)
                seen_normalized.add(q_var4.lower())

    # 2. Intent: Admission Fees
    elif intent in {"admission_fees", "admission_fee", "fees_finance"} or data_type == "fee_structure":
        quota_term = ""
        if quota == "kea_cet":
            quota_term = "KEA KCET government quota"
        elif quota == "comedk":
            quota_term = "COMEDK quota"
        elif quota == "management":
            quota_term = "management quota"

        q_var1 = f"RVCE 1st year B.E. tuition college fees {quota_term}".strip()
        q_var2 = f"RVCE fee circular payment deadline bank challan {quota_term}".strip()
        for q_candidate in [q_var1, q_var2]:
            if q_candidate.lower() not in seen_normalized and len(queries) < max_queries:
                queries.append(q_candidate)
                seen_normalized.add(q_candidate.lower())

    # 3. Intent: Placements
    elif intent in {"placements", "placements_careers"} or company or data_type == "placement_record":
        if company:
            q_var1 = f"RVCE 2027 placement {company} package CTC offers"
            if q_var1.lower() not in seen_normalized and len(queries) < max_queries:
                queries.append(q_var1)
                seen_normalized.add(q_var1.lower())
        if branch:
            q_var2 = f"RVCE 2027 placement statistics {branch} branch average CTC"
            if q_var2.lower() not in seen_normalized and len(queries) < max_queries:
                queries.append(q_var2)
                seen_normalized.add(q_var2.lower())

    # 4. Intent: Academics & Exams
    elif intent in {"academics_exams"} or data_type == "schedule":
        q_var1 = f"RVCE academic calendar timetable examination circular"
        if q_var1.lower() not in seen_normalized and len(queries) < max_queries:
            queries.append(q_var1)
            seen_normalized.add(q_var1.lower())

    # General branch acronym expansion if queries count is low
    if branch and branch in BRANCH_EXPANSIONS and len(queries) < max_queries:
        for alt_name in BRANCH_EXPANSIONS[branch]:
            if alt_name.lower() not in cleaned.lower():
                q_sub = re.sub(rf"\b{re.escape(branch)}\b", alt_name, cleaned, flags=re.IGNORECASE)
                if q_sub.lower() not in seen_normalized:
                    queries.append(q_sub)
                    seen_normalized.add(q_sub.lower())
                    if len(queries) >= max_queries:
                        break

    return queries[:max_queries]


def retrieve_multi_query_hybrid(
    retriever: Any,
    queries: List[str],
    top_k: int = 6,
    score_threshold: float = 0.03,
    strategy: str = "hybrid",
    rerank: bool = False,
    top_k_per_query: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Execute parallel hybrid retrieval across multiple expanded queries and fuse candidates."""
    effective_top_k = top_k_per_query if top_k_per_query is not None else top_k
    if not queries or retriever is None:
        return []

    from src.retrieval import TFIDFRetriever

    all_candidates_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    query_scores_by_key: Dict[Tuple[str, str, str], List[float]] = {}
    query_ranks_by_key: Dict[Tuple[str, str, str], List[int]] = {}

    for q_idx, q_text in enumerate(queries):
        if not q_text.strip():
            continue
        try:
            # Retrieve candidates for this variant
            if hasattr(retriever, "retrieve"):
                if hasattr(retriever, "tfidf_retriever") and hasattr(retriever, "semantic_retriever"):
                    candidates = retriever.retrieve(
                        query=q_text,
                        top_k=max(effective_top_k * 2, 10),
                        score_threshold=0.0,  # collect full candidate pool before fusion
                        strategy=strategy,
                        rerank=False,
                    )
                else:
                    candidates = retriever.retrieve(
                        query=q_text,
                        top_k=max(effective_top_k * 2, 10),
                        score_threshold=0.0,
                    )
            else:
                candidates = []
        except Exception as exc:
            logger.warning("Query retrieval failed for variant '%s': %s", q_text, exc)
            continue

        for rank, cand in enumerate(candidates, start=1):
            key = TFIDFRetriever._dedup_key_for_result(cand)
            score = float(cand.get("relevance_score") or 0.0)

            if key not in all_candidates_by_key:
                cand_copy = dict(cand)
                cand_copy["retrieval_queries"] = [q_text]
                cand_copy["min_rank"] = rank
                all_candidates_by_key[key] = cand_copy
                query_scores_by_key[key] = [score]
                query_ranks_by_key[key] = [rank]
            else:
                if q_text not in all_candidates_by_key[key]["retrieval_queries"]:
                    all_candidates_by_key[key]["retrieval_queries"].append(q_text)
                query_scores_by_key[key].append(score)
                query_ranks_by_key[key].append(rank)
                all_candidates_by_key[key]["min_rank"] = min(all_candidates_by_key[key]["min_rank"], rank)

    # Fuse scores using Reciprocal Rank Fusion (RRF) across query streams
    fused: List[Dict[str, Any]] = []
    rrf_k = 60
    for key, cand in all_candidates_by_key.items():
        ranks = query_ranks_by_key[key]
        scores = query_scores_by_key[key]

        # RRF formula with multi-query frequency boost
        rrf_score = sum(1.0 / (rrf_k + r) for r in ranks)
        max_lexical_semantic_score = max(scores) if scores else 0.0

        # Primary query match bonus
        is_matched_by_primary = len(cand.get("retrieval_queries", [])) > 0 and cand["retrieval_queries"][0] == queries[0]
        primary_bonus = 0.015 if is_matched_by_primary else 0.0

        final_relevance = round(max_lexical_semantic_score * 0.7 + (rrf_score * 10.0) * 0.3 + primary_bonus, 4)
        cand["relevance_score"] = final_relevance
        cand["matched_query_count"] = len(ranks)

        if final_relevance >= score_threshold:
            fused.append(cand)

    fused.sort(key=lambda x: float(x.get("relevance_score") or 0.0), reverse=True)

    # Cross-encoder rerank if enabled
    if rerank and hasattr(retriever, "reranker") and retriever.reranker is not None:
        try:
            fused = retriever.reranker.rerank(queries[0], fused[:max(effective_top_k * 3, 15)])
        except Exception as exc:
            logger.warning("Reranking failed safely: %s", exc)

    return fused[:effective_top_k]
