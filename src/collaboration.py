from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.llm_groq import call_llm

STAGES = ["CLARIFY", "HYPOTHESIZE", "COFIX", "REFLECT"]

def run_stage(stage, learner_input, history, rag):
    # Retrieve
    rag_results = rag.retrieve(learner_input, k=3) if rag else []

    rag_context = "\n".join(
        f"[{doc.doc_id} | score={score:.2f}] {doc.text[:500].replace('\n',' ')}"
        for doc, score in rag_results
    ) if rag_results else "None"

    history_summary = "\n".join(
        f"{h['stage']}: {h['learner'][:200]}"
        for h in history[-4:]
    )

    user_prompt = build_user_prompt(
        stage=stage,
        learner_input=learner_input,
        rag_context=rag_context,
        history_summary=history_summary,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    answer = call_llm(messages)

    # Return both answer + what we retrieved (for UI transparency)
    retrieved = [
        {"doc_id": doc.doc_id, "score": float(score), "excerpt": doc.text[:600]}
        for doc, score in rag_results
    ]

    return answer, retrieved
