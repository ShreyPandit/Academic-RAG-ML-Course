import argparse
import json
import re
from pathlib import Path
from typing import List, Dict
from Retriever.embedding_based import retrieve
from transformers import pipeline

MODEL_NAME = "/data/spandit/Qwen2.5-3B-Instruct"
DEVICE = 0
MAX_NEW_TOKENS_ANS = 1024
MAX_NEW_TOKENS_GRADE = 6


def build_generator(model_name: str = MODEL_NAME, device: int = DEVICE):
    """Return a transformers.pipeline for text generation."""
    return pipeline("text-generation", model=model_name, device_map="auto", trust_remote_code=True)


def generate_answer(question: str, generator_llm) -> str:
    """Generate an answer to *question* using retrieval-augmented context."""
    context_passages = retrieve(question)
    context = "\n\n".join(context_passages)
    # print(f"Context: {context}")
    
    system_prompt = "You are a student who is taking an exam, given a question, and a context, you need to give an answer. Be very to the point, dont say things that are not part of your answer, directly start answering"
    user_prompt = f"You are a student who is taking an exam, given a question, and a context, you need to give an answer. Be very to the point, dont say things that are not part of your answer, directly start answering\n\n Question: {question}\n\n Context:\n{context}\n\n Please give a concise, correct answer"

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    output = generator_llm(prompt, max_new_tokens=MAX_NEW_TOKENS_ANS, do_sample=False)
    # Remove the prompt portion and return only the assistant response
    return output[0]["generated_text"][-1]['content']


def build_grader(model_name: str = MODEL_NAME, device: int = DEVICE):
    """Return a transformers.pipeline used as an LLM judge."""
    return pipeline("text-generation", model=model_name, device_map="auto", trust_remote_code=True)


def grade_answer(question: str, reference: str, student_answer: str, grader_llm) -> int:
    """Ask the grader LLM to give an integer 0-10 score."""
    
    prompt = [
        {"role": "system", "content": "You are an exam grader. You are a linient grader. On a scale of 0-10, where 0 means completely wrong and 10 means perfectly correct, Give marks for steps too.\n"},
        {"role": "user", "content": f"You are a linient grader. On a scale of 0-10, where 0 means completely wrong and 10 means perfectly correct, Give marks for steps too.\nYou are given a question, a reference answer, and a student's answer. score the student's answer.\n\nQuestion: {question}\n\nReference answer: {reference}\n\nStudent answer: {student_answer}\n\nRespond with only the integer score."}
    ]

    result = grader_llm(prompt, max_new_tokens=MAX_NEW_TOKENS_GRADE, do_sample=False)
    match = re.search(r"\d+", result[0]["generated_text"][-1]["content"])
    return int(match.group()) if match else 0


def run_eval(
    qa_pairs: List[Dict[str, str]],
    generator,
    grader,
    output_path: Path | None = None,
):
    """Generate answers and grade them; return total score."""
    results = []
    total = 0
    for i, qa in enumerate(qa_pairs, 1):
        q, ref = qa["question"], qa["answer"]
        student_ans = generate_answer(q, generator)
        # print(f"Q{i}: {q}\nRef: {ref}\nGen: {student_ans}")
        print(f"Generated answer: {student_ans}")
        score = grade_answer(q, ref, student_ans, grader)
        total += score
        results.append(
            {
                "id": i,
                "question": q,
                "reference": ref,
                "generated": student_ans,
                "score": score,
            }
        )
        print(f"Q{i}: {score}/10")
    print(f"\n=== Total: {total}/{len(qa_pairs)*10} ===")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    return total, results

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline + LLM grading")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to JSONL or JSON file with objects: {question:str, answer:str}",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save detailed results as JSON",
    )
    args = parser.parse_args()

    # Load evaluation data
    if args.data.suffix == ".jsonl":
        qa_pairs = [json.loads(line) for line in args.data.read_text().splitlines()]
    else:
        qa_pairs = json.loads(args.data.read_text())

    generator = build_generator()
    grader = build_grader()

    run_eval(qa_pairs, generator, grader, args.save)


if __name__ == "__main__":
    main()
