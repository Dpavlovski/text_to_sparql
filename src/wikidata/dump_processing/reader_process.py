import bz2
import gzip
from multiprocessing import Queue, Value
from pathlib import Path

from tqdm import tqdm


def count_lines(input_file: Path, max_lines_to_read: int):
    cnt = 0
    if input_file.suffix == ".bz2":
        f = bz2.open(input_file, "r")
    elif input_file.suffix == ".gz":
        f = gzip.open(input_file, "rb")
    else:
        raise ValueError(f"The file must be either .bz2 or .gz, but got {input_file.suffix}.")
    for _ in tqdm(f, "Counting lines..."):
        cnt += 1
        if max_lines_to_read > 0 and cnt >= max_lines_to_read:
            break
    return cnt


def read_data(input_file: Path, num_lines_read: Value, max_lines_to_read: int, work_queue: Queue):
    """
    Reads the data from the input file and pushes it to the output queue.
    :param input_file: Path to the input file.
    :param num_lines_read: Value to store the number of lines in the input file.
    :param max_lines_to_read: Maximum number of lines to read from the input file (for testing).
    :param work_queue: Queue to push the data to.
    """
    if input_file.suffix == ".bz2":
        f = bz2.open(input_file, "r")
    elif input_file.suffix == ".gz":
        f = gzip.GzipFile(input_file, "r")
    else:
        raise ValueError(f"The file must be either .bz2 or .gz, but got {input_file.suffix}.")

    num_lines = 0
    for ln in f:
        if ln == b"[\n" or ln == b"]\n":
            continue
        if ln.endswith(b",\n"):  # all but the last element
            obj = ln[:-2]
        else:
            obj = ln
        num_lines += 1
        work_queue.put(obj)
        if 0 < max_lines_to_read <= num_lines:
            break
    num_lines_read.value = num_lines

    f.close()
    return
