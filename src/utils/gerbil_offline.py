import json
import re

# ================= CONFIGURATION =================
GOLD_JSON_PATH = "../../qald_10_with_mk.json"
SYSTEM_JSON_PATH = "../../results/gerbil/sparql_outputs_en_nemotron-3-nano-30b-a3.json"
TARGET_LANG = "en"


def normalize_answer(ans):
    """
    Normalizes URIs and literals to exactly match GERBIL's Java evaluation logic.
    GERBIL natively handles owl:sameAs and strips language tags/datatypes.
    """
    val = str(ans).strip()

    # 1. Normalize HTTP/HTTPS
    if val.startswith("https://"):
        val = "http://" + val[8:]

    # 2. Normalize Wikidata subdomains
    val = val.replace("http://www.wikidata.org/", "http://wikidata.org/")

    # 3. Strip XSD datatypes (e.g., "1999"^^<http://www.w3.org/2001/XMLSchema#date> -> "1999")
    if "^^" in val:
        val = val.split("^^")[0]

    # 4. Strip language tags from literals (e.g., "Berlin"@en -> "Berlin")
    val = re.sub(r'@[a-zA-Z\-]+$', '', val)

    # 5. Remove bounding quotes from literals
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1]

    return val


def extract_answers(qald_item, valid_vars=None):
    """
    Extracts answers from a QALD JSON question block into a normalized set.
    """
    answers = set()
    if "answers" not in qald_item or not qald_item["answers"]:
        return answers

    for ans_obj in qald_item["answers"]:
        # 1. Boolean answers (ASK queries)
        if "boolean" in ans_obj:
            answers.add(str(ans_obj["boolean"]).lower())
            continue

        # 2. SPARQL bindings (SELECT queries)
        if "results" in ans_obj and "bindings" in ans_obj["results"]:
            # If no valid_vars passed, extract them from the current item's head
            if valid_vars is None and "head" in ans_obj and "vars" in ans_obj["head"]:
                valid_vars = set(ans_obj["head"]["vars"])

            for binding in ans_obj["results"]["bindings"]:
                for var_name, var_data in binding.items():
                    # Skip auxiliary variables the system might have returned that aren't requested
                    if valid_vars and var_name not in valid_vars:
                        continue
                    if "value" in var_data:
                        answers.add(normalize_answer(var_data["value"]))

        # 3. Fallback for plain string/number arrays
        if "string" in ans_obj:
            val_data = ans_obj["string"]
            if isinstance(val_data, list):
                for v in val_data:
                    answers.add(normalize_answer(v))
            else:
                answers.add(normalize_answer(val_data))

    return answers


def evaluate_gerbil_kgqa(gold_file, sys_file):
    # Load JSON files
    try:
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        with open(sys_file, 'r', encoding='utf-8') as f:
            sys_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON files: {e}")
        return

    # Handle {"questions": [...]} format vs direct list [...] format
    gold_list = gold_data.get("questions", gold_data) if isinstance(gold_data, dict) else gold_data
    sys_list = sys_data.get("questions", sys_data) if isinstance(sys_data, dict) else sys_data

    # Map ID -> Raw Question Object
    gold_raw_dict = {str(q["id"]): q for q in gold_list}
    sys_raw_dict = {str(q["id"]): q for q in sys_list}

    total_questions = len(gold_raw_dict)
    if total_questions == 0:
        print("Error: No questions found in the Gold Standard file.")
        return

    macro_p_sum, macro_r_sum, macro_f1_sum = 0.0, 0.0, 0.0
    micro_tp, micro_fp, micro_fn = 0, 0, 0

    print("Running GERBIL KGQA offline evaluation...")

    for qid, gold_q in gold_raw_dict.items():
        sys_q = sys_raw_dict.get(qid, {})

        # Determine the exact variables the Gold Standard expects
        gold_vars = None
        if "answers" in gold_q and len(gold_q["answers"]) > 0:
            ans_0 = gold_q["answers"][0]
            if "head" in ans_0 and "vars" in ans_0["head"]:
                gold_vars = set(ans_0["head"]["vars"])

        # Extract normalized sets
        gold_ans = extract_answers(gold_q, valid_vars=gold_vars)
        sys_ans = extract_answers(sys_q, valid_vars=gold_vars)

        # Calculate Absolute Overlaps
        tp_set = gold_ans.intersection(sys_ans)
        tp = len(tp_set)
        fp = len(sys_ans) - tp
        fn = len(gold_ans) - tp

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        # --- OFFICIAL GERBIL/QALD MACRO METRIC RULES ---
        if len(gold_ans) == 0 and len(sys_ans) == 0:
            # Rule 1: Gold is legitimately empty and System outputted nothing -> Perfect score
            p, r, f1 = 1.0, 1.0, 1.0
        else:
            # Rule 2: If system answers nothing (0 length), Precision is 1.0 because it made no mistakes.
            # However, if gold has answers, Recall is 0.0, hence F1 is 0.0.
            p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(sys_ans) == 0 else 0.0)
            r = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(gold_ans) == 0 else 0.0)
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        macro_p_sum += p
        macro_r_sum += r
        macro_f1_sum += f1

    # Calculate final GERBIL-style averages
    macro_p = macro_p_sum / total_questions
    macro_r = macro_r_sum / total_questions
    macro_f1 = macro_f1_sum / total_questions

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Print Final KGQA Report
    print("\n" + "=" * 50)
    print(f"{'GERBIL KGQA OFFLINE REPORT':^50}")
    print("=" * 50)
    print(f" Evaluated Questions : {total_questions}")
    print("-" * 50)
    print(" MACRO METRICS (Official QALD Metric)")
    print(f"   Macro Precision   : {macro_p:.4f}")
    print(f"   Macro Recall      : {macro_r:.4f}")
    print(f"   Macro F1          : {macro_f1:.4f}  <-- (GERBIL Target)")
    print("-" * 50)
    print(" MICRO METRICS (Aggregated Pool)")
    print(f"   Micro Precision   : {micro_p:.4f}")
    print(f"   Micro Recall      : {micro_r:.4f}")
    print(f"   Micro F1          : {micro_f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_gerbil_kgqa(GOLD_JSON_PATH, SYSTEM_JSON_PATH)
