import os
import re
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document

from timer import timer
from log_query_expansion import log_query_expansion


class QueryExpansion():
    def __init__(self, nb_variants: int, temperature: float):
        super().__init__()

        self.nb_variants = nb_variants

        model_name = "gemma-2b-it.Q5_K_M.gguf"
        nb_tokens_per_variant = 25
        max_tokens = self.nb_variants * nb_tokens_per_variant

        self.language_model = LlamaCpp(
            model_path=os.path.join("models", model_name),
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=35,
            n_batch=64,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            verbose=False)


    @timer
    def expand_query(self, query: str, chunks: list[Document]) -> str:

        context = "\n\n".join([chunk.page_content for chunk in chunks])

        prompt = _build_expansion_prompt(
                query=query,
                context=context,
                nb_variants=self.nb_variants)

        alternative_queries_raw = self.language_model.invoke(prompt)

        alternative_queries = _format_llm_output(raw_output=alternative_queries_raw)

        log_message = log_query_expansion(
            query=query,
            chunks=chunks,
            alternative_queries=alternative_queries,
            alternative_queries_raw=alternative_queries_raw,
            prompt=prompt)

        return alternative_queries, log_message


def _build_expansion_prompt(query: str, context: str, nb_variants: int) -> str:

    instructions = f"""
        Your role:
        - You are a query expansion engine.

        Your task:
        - Generate {nb_variants} alternative queries by rephrasing the user query.
        - The alternative queries must have the same meaning as the user query.
        - Do not ask about anything diverging from the user query.
        """

    prompt = f"""
        {instructions}\n\n
        User query: {query}\n\n
        Context: {context}\n\n
        """

    return prompt


def _format_llm_output(raw_output: str) -> list[str]:

    pattern_a = re.compile(r'\*\*Alternative Query \s*\d+:\*\*\s*(.*)', re.MULTILINE)
    output_a = pattern_a.findall(raw_output)

    pattern_b = re.compile(r'^\s*\d+\.\s*(.*)', re.MULTILINE)
    output_b = pattern_b.findall(raw_output)

    clean_output = [output_ab.strip() for output_ab in (output_a + output_b) if output_ab.strip()]

    return clean_output
