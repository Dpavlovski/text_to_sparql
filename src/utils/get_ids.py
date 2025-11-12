import csv
import json
import re


def extract_ids_from_sparql(json_file_path, csv_file_path):
    """
    Extracts a question's ID and all Wikidata IDs (e.g., Q123, P456)
    from its SPARQL query in a JSON file, then saves the results to a CSV.

    Args:
        json_file_path (str): The path to the input JSON file.
        csv_file_path (str): The path for the output CSV file.
    """
    try:
        # Load the JSON data from the file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # This list will hold all the rows for our CSV file
        csv_output_data = []

        # Define the regular expression to find all P and Q identifiers
        # \b ensures we match whole words only (e.g., avoids matching 'Q1' in 'ABCQ12')
        id_pattern = re.compile(r'\b([PQ]\d+)\b')

        if 'questions' in data:
            for question in data['questions']:
                question_id = question.get('id')
                sparql_query = question.get('query', {}).get('sparql')

                if question_id is not None and sparql_query is not None:
                    # Find all unique IDs in the SPARQL query string
                    found_ids = sorted(list(set(id_pattern.findall(sparql_query))))

                    # Create a row with the question ID and all found SPARQL IDs
                    row = [question_id] + found_ids
                    csv_output_data.append(row)

        # Find the maximum number of IDs found in any single query to create headers
        max_ids = 0
        if csv_output_data:
            max_ids = max(len(row) for row in csv_output_data) - 1

        # Create dynamic headers: ['Question_ID', 'ID_1', 'ID_2', ...]
        headers = ['Question_ID'] + [f'ID_{i + 1}' for i in range(max_ids)]

        # Write the extracted data to a CSV file
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(csv_output_data)

        print(f"✅ Success! Extracted data for {len(csv_output_data)} questions to '{csv_file_path}'")

    except FileNotFoundError:
        print(f"❌ Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from the file '{json_file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


json_filename = '../../qald_10_with_mk.json'
csv_filename = 'extracted_sparql_ids.csv'

extract_ids_from_sparql(json_filename, csv_filename)
