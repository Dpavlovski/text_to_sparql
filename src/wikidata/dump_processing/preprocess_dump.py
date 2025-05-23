import argparse
import multiprocessing
import time
from multiprocessing import Queue, Process
from pathlib import Path

from src.wikidata.dump_processing.reader_process import count_lines, read_data
from src.wikidata.dump_processing.worker_process import process_data
from src.wikidata.dump_processing.writer_process import write_data


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='path to bz2 wikidata json dump')
    parser.add_argument('--out_dir', type=str, required=True, help='path to output directory')
    parser.add_argument('--processes', type=int, default=90, help="number of concurrent processes to spin off. ")
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_lines_read', type=int, default=-1,
                        help='Terminate after num_lines_read lines are read. Useful for debugging.')
    parser.add_argument('--num_lines_in_dump', type=int, default=-1,
                        help='Number of lines in dump. If -1, we will count the number of lines.')
    return parser


def main(args):
    start = time.time()
    print(f"ARGS: {args}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    input_file = Path(args.input_file)
    assert input_file.exists(), f"Input file {input_file} does not exist"

    max_lines_to_read = args.num_lines_read
    if args.num_lines_in_dump <= 0:
        print("Counting lines")
        total_num_lines = count_lines(input_file, max_lines_to_read)
    else:
        total_num_lines = args.num_lines_in_dump

    print("Starting processes")
    maxsize = 10 * args.processes
    output_queue = Queue(maxsize=maxsize)
    work_queue = Queue(maxsize=maxsize)

    num_lines_read = multiprocessing.Value("i", 0)
    read_process = Process(
        target=read_data,
        args=(input_file, num_lines_read, max_lines_to_read, work_queue)
    )

    read_process.start()

    write_process = Process(
        target=write_data,
        args=(out_dir, args.batch_size, total_num_lines, output_queue)
    )
    write_process.start()

    work_processes = []
    for _ in range(max(1, args.processes - 2)):
        work_process = Process(
            target=process_data,
            args=(work_queue, output_queue)
        )
        work_process.daemon = True
        work_process.start()
        work_processes.append(work_process)

    read_process.join()
    print(f"Done! Read {num_lines_read.value} lines")

    for work_process in work_processes:
        work_queue.put(None)

    for work_process in work_processes:
        work_process.join()

    output_queue.put(None)
    write_process.join()

    print(f"Finished processing {num_lines_read.value} lines in {time.time() - start}s")


if __name__ == "__main__":
    args = get_arg_parser().parse_args([
        "--input_file", "../../../wikidata_data/latest-all.json.bz2",
        "--out_dir", "data_processed",
        "--processes", "32",
        "--batch_size", "300",
        "--num_lines_read", "1000",
        "--num_lines_in_dump", "115188728"
    ])

    main(args)
