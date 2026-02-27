from src.utils.format_uri import extract_id_from_uri


def extract_all_qids(data):
    """Recursively finds all strings looking like Q-IDs in a nested JSON object."""
    qids = set()

    if isinstance(data, dict):
        for key, value in data.items():
            # Optimization: In SPARQL JSON results, the valid QID is usually
            # inside a dict where "type" is "uri" and the key is "value".
            if key == "value" and isinstance(value, str):
                # Only try to extract if it looks like a Wikidata URI or starts with Q/P
                if "wikidata.org/entity/" in value or value.startswith("Q") or value.startswith("P"):
                    qid = extract_id_from_uri(value)
                    if qid: qids.add(qid)
            else:
                qids.update(extract_all_qids(value))

    elif isinstance(data, list):
        for item in data:
            qids.update(extract_all_qids(item))

    elif isinstance(data, str):
        # Only try if string looks promising
        if "wikidata.org" in data or (len(data) < 15 and (data.startswith("Q") or data.startswith("P"))):
            qid = extract_id_from_uri(data)
            if qid: qids.add(qid)

    return qids
