from datasets import load_dataset, concatenate_datasets


def get_dataset():
    dataset = load_dataset('mohnish/lc_quad', trust_remote_code=True)

    print(dataset.keys())

    train_dataset = dataset['train']

    test_dataset = dataset['test']

    combined_dataset = concatenate_datasets([train_dataset, test_dataset])

    return combined_dataset
