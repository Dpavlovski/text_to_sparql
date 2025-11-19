from typing import Dict, List, Any


def format_candidates_for_prompt(candidates_map: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Formats the candidate entities dictionary into a clean, structured string
    for the LLM prompt.

    Args:
        candidates_map: Dictionary mapping keywords to lists of candidate entities.
                        e.g., {'keyword': [{'id': 'Q1', 'label': 'L', 'description': 'D'}, ...]}

    Returns:
        A formatted string listing candidates for each keyword.
    """
    if not candidates_map:
        return "No candidate entities found."

    formatted_output = []

    for keyword, candidates in candidates_map.items():
        # Add a header for the keyword
        section = f"For the keyword '{keyword}':"

        if not candidates:
            section += "\n  - No matching candidates found in Wikidata."
        else:
            # Limit to top 5 candidates to save tokens and reduce noise
            for i, cand in enumerate(candidates[:5], 1):
                qid = cand.get('id', 'Unknown ID')
                label = cand.get('label', 'No label')
                description = cand.get('description', 'No description')

                # Format: "1. [Q123] Label - Description"
                line = f"  {i}. [{qid}] {label} - {description}"
                section += f"\n{line}"

        formatted_output.append(section)

    # Join all sections with a double newline for separation
    return "\n\n".join(formatted_output)
