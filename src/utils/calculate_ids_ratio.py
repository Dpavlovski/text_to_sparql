# import pandas as pd
#
# # Load the datasets
# df_sparql = pd.read_csv('../../results/benchmark/sparql_outputs_mk.csv')
# df_ids = pd.read_csv('../../ground_truth_ids.csv')
#
# # Get the unique questions from the sparql_outputs_v3.csv file in order of appearance
# unique_questions = df_sparql['original_question'].unique()
#
# # Create a mapping from each unique question to the corresponding Combined_ID
# question_to_id_map = dict(zip(unique_questions, df_ids['Combined_IDs']))
#
# # Map the Combined_IDs to the original_question column in the sparql_outputs_v3.csv dataframe
# df_sparql['Combined_IDs'] = df_sparql['original_question'].map(question_to_id_map)
#
# # Save the updated dataframe to a new CSV file
# df_sparql.to_csv('../../results/benchmark/sparql_outputs_mk_2.csv', index=False)
#
# print("IDs have been successfully inserted into the 'sparql_outputs_with_ids.csv' file.")


import re

import pandas as pd


def create_analysis_file(input_path, output_path):
    """
    Reads a CSV, analyzes Wikidata ID matches, adds new columns for the
    extracted candidates and the match ratio, and saves it to a new file.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path where the output CSV file will be saved.
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_path)
        print(f"Successfully loaded the dataset from '{input_path}'.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {input_path}")
        return

    # --- Step 1: Extract Candidate and Ground Truth IDs ---

    def extract_wikidata_ids(candidates_str):
        """Extracts Wikidata IDs, handling both 'id' and 'qid' keys."""
        try:
            return re.findall(r"'(?:id|qid)': '(Q\d+|P\d+)'", str(candidates_str))
        except:
            return []

    def extract_ground_truth_ids(combined_ids_str):
        """Extracts Wikidata Q and P identifiers from the 'Combined_IDs' column."""
        try:
            return re.findall(r"'(Q\d+|P\d+)'", str(combined_ids_str))
        except:
            return []

    print("Extracting candidate and ground truth IDs...")
    df['wikidata_candidate_ids'] = df['candidates'].apply(extract_wikidata_ids)
    df['ground_truth_wikidata_ids'] = df['Combined_IDs'].apply(extract_ground_truth_ids)

    # --- Step 2: Calculate Match Ratio ---

    def calculate_match_ratio(row):
        """
        Calculates the ratio of matching IDs to total ground truth IDs
        and returns it as a string (e.g., '1/3').
        """
        ground_truth_ids = row['ground_truth_wikidata_ids']
        candidate_ids = row['wikidata_candidate_ids']

        # Avoid division by zero if there are no ground truth IDs
        if not ground_truth_ids:
            return "0/0"

        # Find the number of common IDs
        matches = len(set(ground_truth_ids).intersection(set(candidate_ids)))
        total_ground_truth = len(ground_truth_ids)

        return f"{matches}/{total_ground_truth}"

    print("Calculating match ratios...")
    # Add the new 'match_ratio' column
    df['match_ratio'] = df.apply(calculate_match_ratio, axis=1)

    # --- Step 3: Save the New File ---

    try:
        # Select columns to save. We'll keep the original ones and add the new ones.
        # Let's place the new columns near the beginning for easy viewing.
        output_columns = [
                             'original_question',
                             'wikidata_candidate_ids',
                             'ground_truth_wikidata_ids',
                             'match_ratio'
                         ] + [col for col in df.columns if col not in [
            'original_question', 'wikidata_candidate_ids', 'ground_truth_wikidata_ids', 'match_ratio'
        ]]

        df_to_save = df[output_columns]

        # Save the DataFrame to a new CSV file
        df_to_save.to_csv(output_path, index=False)
        print(f"\n--- Analysis Complete ---")
        print(f"Successfully saved the new file to: {output_path}")

    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == '__main__':
    # Define the input and output file paths
    input_csv = '../../results/benchmark/sparql_outputs_with_analysis.csv'
    output_csv = '../../results/benchmark/sparql_outputs_with_analysis_2.csv'

    # Run the function to create the new analysis file
    create_analysis_file(input_csv, output_csv)
