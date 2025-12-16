from datasets import load_dataset, concatenate_datasets
from transformers import pipeline

# 1. CHANGED: Model updated to English -> Russian (ru)
try:
    TRANSLATOR = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-ru",
        device=-1  # Use 0 if you have a GPU, -1 for CPU
    )
    print("Russian translation model loaded successfully.")
except Exception as e:
    print(f"Error loading translation model: {e}")
    TRANSLATOR = None


def get_dataset():
    dataset = load_dataset('mohnish/lc_quad', trust_remote_code=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    combined_dataset = concatenate_datasets([train_dataset, test_dataset])
    return combined_dataset


# 2. CHANGED: Function name and logic for Russian
def translate_batch_to_ru(questions: list[str]) -> list[str]:
    """
    Translates a batch of English questions to Russian.
    """
    if not TRANSLATOR:
        return [f"ERROR: MODEL NOT LOADED. ORIGINAL: {q}" for q in questions]

    valid_questions = []
    valid_indices = []

    for i, q in enumerate(questions):
        if isinstance(q, str) and q.strip():
            valid_questions.append(q)
            valid_indices.append(i)

    full_output = [''] * len(questions)

    if valid_questions:
        # 3. Translation execution
        results = TRANSLATOR(valid_questions, max_length=256, batch_size=16, truncation=True)
        translated_texts = [item['translation_text'] for item in results]

        for original_index, translated_text in zip(valid_indices, translated_texts):
            full_output[original_index] = translated_text

    return full_output


def get_translated_and_embedded_dataset():
    dataset = get_dataset()

    print("Starting translation to Russian in batches...")

    # 4. CHANGED: Column name to 'question_ru'
    translated_dataset = dataset.map(
        lambda x: {'question_ru': translate_batch_to_ru(x['question'])},
        batched=True,
        batch_size=32
    )
    print("Translation complete.")

    # 5. CHANGED: Save path to lcquad2_ru
    output_path = './lcquad2_ru'
    translated_dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

    return translated_dataset


if __name__ == '__main__':
    final_dataset = get_translated_and_embedded_dataset()
    print("\nFinal Dataset Columns:", final_dataset.column_names)

    print("\n--- SAMPLE OF TRANSLATED DATASET (First 3 Rows) ---")
    # Prints the English question alongside the Russian translation
    print(final_dataset.select(range(3)).to_pandas()[['question', 'question_ru', 'sparql_wikidata']])
