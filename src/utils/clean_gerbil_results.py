import json

# ================= КОНФИГУРАЦИЈА =================
# Патеката до твојот "валкан" JSON фајл
INPUT_JSON = "../../results/gerbil/en_opus_4.7_zero.json"

# Каде да се зачува исчистениот фајл
OUTPUT_JSON = "../../results/gerbil/en_opus_4.7_zero_clean.json"


# =================================================

def clean_gerbil_results(input_path, output_path):
    print(f"Вчитување на податоци од {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # GERBIL фајловите може да почнуваат со "questions" или директно со листа
    questions = data.get("questions", data) if isinstance(data, dict) else data

    fixed_count = 0

    for q in questions:
        for answer in q.get("answers", []):
            if "results" in answer and "bindings" in answer["results"]:
                for binding in answer["results"]["bindings"]:
                    # Провери секоја променлива во binding-от (најчесто е "result", "uri" итн.)
                    for var_name, var_data in binding.items():
                        if "value" in var_data:
                            val = var_data["value"]

                            # Ако вредноста почнува со Wikidata URI и има празно место во неа
                            if val.startswith("http://www.wikidata.org/entity/") and " " in val:
                                # Отсечи сè после првото празно место
                                clean_val = val.split(" ")[0]
                                var_data["value"] = clean_val
                                fixed_count += 1

    # Зачувај го исчистениот фајл
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Успешно поправени {fixed_count} резултати!")
    print(f"Новиот фајл е зачуван во: {output_path}")


if __name__ == "__main__":
    clean_gerbil_results(INPUT_JSON, OUTPUT_JSON)
