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


def generate_answer(question: str, generator_llm, retriever: str) -> str:
    """Generate an answer to *question* using retrieval-augmented context."""
    ## Using RAG
    if retriever is not 'none':
        context_passages = retrieve(question)
        context = "\n\n".join(context_passages)
        # print(f"Context: {context}")
        system_prompt = "You are a student taking an exam. For each question, you are provided with a context. Your task is to answer the question directly and concisely. Do not include any extra commentary or introductions — immediately provide the answer with help of the given context."
        user_prompt = f"You are a student taking an exam. You will be given a question and a context. Your task is to answer the question directly and concisely, using the information provided. Do not include any extra commentary, explanations, or introductions — begin your answer immediately. \nQuestion: {question}\nContext: {context}\nPlease provide a concise and accurate answer."
    
    ## Without using RAG
    else:
        system_prompt = "You are a student taking an exam. Your task is to answer the question directly and concisely. Do not include any extra commentary or introductions — immediately provide the answer."
        user_prompt = f"You are a student taking an exam. You will be given a question. Your task is to answer the question directly and concisely. Do not include any extra commentary, explanations, or introductions — begin your answer immediately. \nQuestion: {question}\nPlease provide a concise and accurate answer."
        
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
        {"role": "system", "content": "You are an exam grader. You grade leniently. For each answer, assign a score on a scale from 0 to 10, where: 0 means completely incorrect, and 10 means perfectly correct. Give credit for partial steps and partial understanding wherever possible."},
        {"role": "user", "content": f"You are a lenient grader. Score the student's answer on a scale from 0 to 10, where: 0 means completely incorrect, and 10 means perfectly correct. Give credit for partial steps and partial understanding wherever applicable. You will be provided with a question, a reference answer, and a student's answer. Evaluate the student's answer and respond with only the integer score (no explanations). \nQuestion: {question}\nReference Answer: {reference}\nStudent Answer: {student_answer}"}
    ]

    result = grader_llm(prompt, max_new_tokens=MAX_NEW_TOKENS_GRADE, do_sample=False)
    match = re.search(r"\d+", result[0]["generated_text"][-1]["content"])
    return int(match.group()) if match else 0


def run_eval(
    qa_pairs: List[Dict[str, str]],
    generator,
    retriever,
    grader,
    output_path: Path | None = None,
):
    """Generate answers and grade them; return total score."""
    results = []
    total = 0
    for i, qa in enumerate(qa_pairs, 1):
        q, ref = qa["question"], qa["answer"]
        student_ans = generate_answer(q, generator, retriever)
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
    parser.add_argument(
        "--retriever",
        type=str,
        default='embed',
        help="Which type of retriever to use. Can be one of 'none' (for no RAG), 'embed', and 'bm25'.",
    )
    args = parser.parse_args()

    # Load evaluation data
    if args.data.suffix == ".jsonl":
        qa_pairs = [json.loads(line) for line in args.data.read_text().splitlines()]
    else:
        qa_pairs = json.loads(args.data.read_text())

    generator = build_generator()
    retriever = args.retriever
    grader = build_grader()

    run_eval(qa_pairs, generator, retriever, grader, args.save)


if __name__ == "__main__":
    main()
