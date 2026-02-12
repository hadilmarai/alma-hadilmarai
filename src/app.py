import os
from dotenv import load_dotenv
from groq import Groq

from rag import MiniRAG
from prompts import SYSTEM_PROMPT, build_user_prompt
from logger import JSONLLogger

STAGES = ["CLARIFY", "HYPOTHESIZE", "COFIX", "REFLECT"]

def format_rag(results):
    lines = []
    for doc, score in results:
        excerpt = doc.text[:800].replace("\n", " ").strip()
        lines.append(f"[{doc.doc_id} | score={score:.2f}] {excerpt}")
    return "\n".join(lines)

def main():
    load_dotenv()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.getenv("MODEL", "openai/gpt-oss-120b")

    rag = MiniRAG.from_folder("data")
    logger = JSONLLogger("logs.jsonl")

    print("ALMA Collaborative LLM Prototype (CLI)")
    print("Paste your problem/code + error/test case. Type 'quit' to exit.\n")

    history_summary = ""
    stage_idx = 0

    while True:
        learner_input = input("\nYOU> ").strip()
        if learner_input.lower() in {"quit", "exit"}:
            break

        stage = STAGES[stage_idx % len(STAGES)]
        rag_results = rag.retrieve(learner_input, k=3)
        rag_context = format_rag(rag_results)

        user_prompt = build_user_prompt(stage, learner_input, rag_context, history_summary)

        logger.log("turn_start", {"stage": stage, "learner_input": learner_input})

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )

        assistant_msg = resp.choices[0].message.content

        assistant_msg = resp.choices[0].message.content
        print(f"\nLLM ({stage})> {assistant_msg}")

        # Super-light "summary" for next turn (kept minimal on purpose)
        history_summary = (history_summary + f"\n[{stage}] Learner: {learner_input}\nLLM: {assistant_msg}\n")[-2000:]

        logger.log("turn_end", {"stage": stage, "assistant_msg": assistant_msg})
        stage_idx += 1

if __name__ == "__main__":
    main()
