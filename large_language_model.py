import os
from langchain_mistralai import ChatMistralAI
from langchain.messages import AIMessage
import json
import ast

from timer import sync_timer


class LargeLanguageModel():
    def __init__(self, temperature: float):
        super().__init__()

        os.environ.get("MISTRAL_API_KEY")

        model_name = "magistral-medium-2506"

        self.language_model = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            max_retries=0)


    @sync_timer
    def get_answer_from_query(self, query: str, context: str) -> str:
        prompt = _build_answering_prompt(query=query, context=context)
        raw_answer = self.language_model.invoke(prompt)
        answer = _get_answer_from_raw_llm_output(raw_output=raw_answer)
        return answer


def _build_answering_prompt(query: str, context: str) -> str:

    instructions = f"""
        Your role:
        - You are an expert Accurate Document Analyst and Knowledge Extractor.
        - Your sole purpose is to provide precise, factual, and fully grounded answers.

        Your task:
        - Analyze the provided `query` and the `context` text.
        - Generate a comprehensive, direct, and accurate answer to the query.
        - Your entire answer must be sourced only from the content found within the `context` block.
        - Do not use any outside knowledge, assumptions, or reasoning not explicitly supported by the `context` block.
        - If the context does not contain the information necessary to answer the query, you must respond with: "I cannot find the answer to this question in the provided document."

        Formatting:
        - You must respond with a single, clean JSON markdown block. DO NOT include any text, reasoning, conversation, or markdown before or after the JSON block.
        """

    prompt = f"""
        {instructions}\n\n
        User query: {query}\n\n
        Context: {context}\n\n
        """

    return prompt


def _get_answer_from_raw_llm_output(raw_output: AIMessage) -> str:

    raw_output = raw_output.content

    raw_output = str(raw_output)

    raw_output = ast.literal_eval(raw_output)

    final_output = raw_output[-1]

    final_text = final_output['text']

    stripped_text = final_text.strip()

    if stripped_text.startswith("```json"):
        stripped_text = stripped_text[len("```json"):].strip()

    if stripped_text.endswith("```"):
        stripped_text = stripped_text[:-len("```"):].strip()

    answer_field = json.loads(stripped_text)

    answer = answer_field["answer"]

    return answer
