import os
import re
from langchain_community.llms import LlamaCpp

from timer import timer


def build_expansion_prompt(user_query: str, context: str, nb_variants: int) -> str:

    instructions = f"""
        Your role:
        - You are a query expansion engine.

        Your task:
        - Generate {nb_variants} alternative queries.
        - Only generate queries that have the same meaning as the user query.
        """

    prompt = f"""
        {instructions}\n\n
        User query: {user_query}\n\n
        Context: {context}\n\n
        """

    return prompt


def format_llm_output(raw_output: str) -> str:
    clean_output = [raw_output.strip()]
    clean_output = re.split(r"\*\*Query \d+:\*\*", clean_output[0], flags=re.IGNORECASE)
    clean_output = [q.strip() for q in clean_output if q.strip()]
    return clean_output


@timer
def expand_query(language_model, user_query: str, nb_variants: int, context: str):
    prompt = build_expansion_prompt(
                user_query=user_query,
                nb_variants=nb_variants,
                context=context)
    return language_model.invoke(prompt)


if __name__ == "__main__":

    user_query = "what is a spell?"

    model_name = "gemma-2b-it.Q5_K_M.gguf"
    nb_variants = 3
    nb_tokens_per_variant = 25
    max_tokens = nb_variants * nb_tokens_per_variant

    language_model = LlamaCpp(
        model_path=os.path.join("models", model_name),
        n_ctx=8192,
        n_threads=8,
        n_gpu_layers=35,
        n_batch=64,
        temperature=0.0,
        top_p=0.9,
        max_tokens=max_tokens,
        verbose=False)

    with open(os.path.join("data", "mtg.txt"), "r") as fid:
        context = fid.read()

    alternative_queries_raw = expand_query(language_model, user_query, nb_variants, context=context)

    alternative_queries = format_llm_output(raw_output=alternative_queries_raw)

    print("\nUser query:")
    print(user_query)
    print("\nAlternative queries:")
    for q in range(len(alternative_queries)):
        print("[{}] {}".format(q, alternative_queries[q]))
