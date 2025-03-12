import json


def load_qald_results() -> list[str]:
    with open("C:\\Users\\User\PycharmProjects\\text_to_sparql\\src\\dataset\\qald_10.json", "r",
              encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    expected_results = []
    for question in questions:

        if 'boolean' in question['answers'][0]:
            continue
        elif 'results' in question['answers'][0]:
            results = question['answers'][0]['results']
            if 'bindings' in results and results['bindings']:
                expected_result = [
                    binding['result']['value']
                    for binding in results['bindings']
                    if 'result' in binding and 'value' in binding['result']
                ]
            else:
                continue
        else:
            continue
        expected_results.extend(expected_result)

    filtered_results = [r for r in expected_results if r.startswith("http://www.wikidata.org")]

    return filtered_results
