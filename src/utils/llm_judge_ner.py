from typing import List

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.llm.llm_provider import LLMProvider
from utils.outputs_analysis import SPARQLUtils
from wikidata.api import get_wikidata_labels

INPUT_CSV = "../../results/benchmark/en_opus_4.7_full_analyzed.csv"
OUTPUT_CSV = "../../results/benchmark/with_neighbors/processed/en_opus_4.7_full_analyzed_judged.csv"
NER_COLUMN_NAME = "ner"


class ConceptMatch(BaseModel):
    gold_label: str = Field(description="The canonical label required by the Gold SPARQL (e.g., 'taxon').")
    matched_agent_keyword: str = Field(description="The EXACT keyword extracted by the Agent. 'NONE' if no match.")
    is_semantic_match: bool = Field(description="True if the keyword means the same thing logically as the gold_label.")


class NEREvaluation(BaseModel):
    clean_agent_keywords: List[str] = Field(description="Clean list of ONLY the keywords.")
    concept_mappings: List[ConceptMatch] = Field(description="Mappings from Gold Label to agent keywords.")
    hallucinated_keywords: List[str] = Field(description="Keywords that do NOT map to any required Gold Label.")
    judge_reasoning: str = Field(description="A brief 1-sentence justification.")


llm = LLMProvider.get_model("gpt-4.1-mini")
structured_llm = llm.with_structured_output(NEREvaluation.model_json_schema())

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial, expert evaluator grading a Named Entity Recognition (NER) system..."),
    ("human",
     'User Question: "{question}"\nGold Labels: {gold_labels}\nAgent Extracted: "{ner_string}"\nMap the concepts.')
])

judge_chain = judge_prompt | structured_llm


def evaluate_ner():
    logger.info(f"Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    df['judge_gold_entities'] = ""
    df['judge_agent_entities'] = ""
    df['ner_precision'], df['ner_recall'], df['ner_f1'] = 0.0, 0.0, 0.0
    df['judge_reasoning'] = ""

    total_evaluated, sum_p, sum_r, sum_f1 = 0, 0.0, 0.0, 0.0

    logger.info("Starting Semantic LLM-as-a-Judge Evaluation...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = str(row.get('original_question', ''))
        gold_sparql = str(row.get('gold_query', ''))
        ner_string = str(row.get(NER_COLUMN_NAME, ''))

        if pd.isna(gold_sparql) or not gold_sparql.strip() or pd.isna(ner_string) or not ner_string.strip():
            continue

        gold_ids = SPARQLUtils.extract_ids_from_text(gold_sparql, ignore_structural=True)

        if gold_ids:
            labels_dict = get_wikidata_labels(list(gold_ids), language="en")

            formatted_labels = []
            for qid, label in labels_dict.items():
                formatted_labels.append(f"'{label}' ({qid})")

            gold_labels_context = ", ".join(formatted_labels)
        else:
            gold_labels_context = "No entities or properties required."

        df.at[index, 'judge_gold_entities'] = gold_labels_context
        df.at[index, 'judge_agent_entities'] = ner_string
        try:
            raw_dict = judge_chain.invoke({
                "question": question, "gold_labels": gold_labels_context, "ner_string": ner_string
            })

            result = NEREvaluation(**raw_dict)

            true_positives = sum(1 for m in result.concept_mappings if m.is_semantic_match)
            false_negatives = sum(1 for m in result.concept_mappings if not m.is_semantic_match)
            false_positives = len(result.hallucinated_keywords)

            total_extracted = true_positives + false_positives
            total_gold_required = true_positives + false_negatives

            p = true_positives / total_extracted if total_extracted > 0 else 0.0
            r = true_positives / total_gold_required if total_gold_required > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            df.at[index, 'ner_precision'] = p
            df.at[index, 'ner_recall'] = r
            df.at[index, 'ner_f1'] = f1
            df.at[index, 'judge_reasoning'] = result.judge_reasoning

            total_evaluated += 1
            sum_p += p
            sum_r += r
            sum_f1 += f1

        except Exception as e:
            logger.warning(f"Error evaluating row {index}: {e}")

    df.to_csv(OUTPUT_CSV, index=False)

    if total_evaluated > 0:
        logger.success("Semantic NER LLM-as-a-Judge Results")
        logger.info(f"Macro Precision: {(sum_p / total_evaluated) * 100:.1f}%")
        logger.info(f"Macro Recall:    {(sum_r / total_evaluated) * 100:.1f}%")
        logger.info(f"Macro F1 Score:  {(sum_f1 / total_evaluated) * 100:.1f}%")


if __name__ == "__main__":
    evaluate_ner()
