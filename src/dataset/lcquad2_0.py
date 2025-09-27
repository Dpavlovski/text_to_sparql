from datasets import load_dataset, concatenate_datasets
from transformers import pipeline

try:
    TRANSLATOR = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-mk",
        device=-1
    )
    print("Macedonian translation model loaded successfully.")
except Exception as e:
    print(f"Error loading translation model: {e}")
    TRANSLATOR = None


def get_dataset():
    dataset = load_dataset('mohnish/lc_quad', trust_remote_code=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    combined_dataset = concatenate_datasets([train_dataset, test_dataset])
    return combined_dataset


def translate_batch_to_mkd(questions: list[str]) -> list[str]:
    """
    Translates a batch of English questions to Macedonian.

    This function is updated to safely handle None, empty, or non-string
    values in the batch, which caused the ValueError.
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
        results = TRANSLATOR(valid_questions, max_length=256, batch_size=16, truncation=True)
        translated_texts = [item['translation_text'] for item in results]

        for original_index, translated_text in zip(valid_indices, translated_texts):
            full_output[original_index] = translated_text

    return full_output


def get_translated_and_embedded_dataset():
    dataset = get_dataset()

    print("Starting translation to Macedonian in batches...")

    translated_dataset = dataset.map(
        lambda x: {'question_mkd': translate_batch_to_mkd(x['question'])},
        batched=True,
        batch_size=32
    )
    print("Translation complete.")

    translated_dataset.save_to_disk('./lcquad2_mk')

    return translated_dataset


if __name__ == '__main__':
    final_dataset = get_translated_and_embedded_dataset()
    print("\nFinal Dataset Columns:", final_dataset.column_names)

    print("\n--- SAMPLE OF TRANSLATED DATASET (First 3 Rows) ---")
    print(final_dataset.select(range(3)).to_pandas())
