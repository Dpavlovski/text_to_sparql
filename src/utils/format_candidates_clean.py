from typing import List, Dict, Any


def format_candidates_clean(candidates_map: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Formats candidates into a clean list.
    - Removes global duplicates (if an ID appeared in a previous mention, it is NOT shown again).
    - Hides description if missing.
    - Shows Graph Context if available.
    """
    if not candidates_map:
        return "No candidates found."

    output = []
    global_described_ids = set()

    for mention, candidates in candidates_map.items():
        # Buffer the output for this mention so we only write the header if there are valid new candidates
        mention_output = []
        mention_output.append(f"**Mention:** '{mention}'")

        has_new_candidates = False
        count = 0

        if not candidates:
            mention_output.append("  (No candidates found)")
        else:
            for c in candidates:
                c_id = c.get('id', '')
                if not c_id: continue

                if c_id in global_described_ids:
                    continue

                global_described_ids.add(c_id)
                has_new_candidates = True
                count += 1

                label = c.get('label', 'No Label')
                desc = c.get('description', '')

                # 2. Format Description: Only show if valid
                if desc and desc.lower() not in ['no description available', 'no description', 'none']:
                    line = f"  {count}. [{c_id}] {label} ({desc})"
                else:
                    line = f"  {count}. [{c_id}] {label}"

                mention_output.append(line)

                # 3. Append Neighbors (Graph Context)
                neighbors = c.get('neighbors', [])
                if neighbors:
                    mention_output.append("     *Graph Context:*")
                    for n in neighbors:
                        mention_output.append(f"     {n}")

        # Only append this block if it actually contributed new information
        if has_new_candidates or not candidates:
            output.extend(mention_output)
            output.append("")  # Spacer

    return "\n".join(output).strip()
