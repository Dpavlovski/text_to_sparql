import requests
from tqdm import tqdm


def download_wikidata_json_dump(url: str, output_file: str) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192  # bytes

    with open(output_file, 'wb') as file, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=output_file
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    print("Download complete!")


url = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2"
output_file = "latest-all.json.bz2"
download_wikidata_json_dump(url, output_file)
