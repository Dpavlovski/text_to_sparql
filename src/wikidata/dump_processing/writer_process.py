import shutil
import time
from multiprocessing import Queue
from pathlib import Path
from typing import Dict, Any, List

import ujson

TABLE_NAMES = [
    'labels', 'descriptions'
]


class Table:
    def __init__(self, path: Path, batch_size: int, table_name: str):
        self.table_dir = path / table_name
        if self.table_dir.exists():
            shutil.rmtree(self.table_dir)
        self.table_dir.mkdir(parents=True, exist_ok=False)

        self.index = 0
        self.cur_num_lines = 0
        self.batch_size = batch_size
        self.cur_file = self.table_dir / f"{self.index:d}.jsonl"
        self.cur_file_writer = None

    def write(self, json_value: List[Dict[str, Any]]):
        if self.cur_file_writer is None:
            self.cur_file_writer = open(self.cur_file, 'w', encoding='utf-8')
        for json_obj in json_value:
            self.cur_file_writer.write(ujson.dumps(json_obj, ensure_ascii=False) + '\n')
        self.cur_num_lines += 1
        if self.cur_num_lines >= self.batch_size:
            self.cur_file_writer.close()
            self.cur_num_lines = 0
            self.index += 1
            self.cur_file = self.table_dir / f"{self.index:d}.jsonl"
            self.cur_file_writer = None

    def close(self):
        self.cur_file_writer.close()


class Writer:
    def __init__(self, path: Path, batch_size: int, total_num_lines: int):
        self.cur_num_lines = 0
        self.total_num_lines = total_num_lines
        self.start_time = time.time()
        self.output_tables = {table_name: Table(path, batch_size, table_name) for table_name in TABLE_NAMES}

    def write(self, json_object: Dict[str, Any]):
        self.cur_num_lines += 1
        for key, value in json_object.items():
            if key not in self.output_tables:
                continue
            if len(value) > 0:
                self.output_tables[key].write(value)
        if self.cur_num_lines % 200000 == 0:
            time_elapsed = time.time() - self.start_time
            estimated_time = time_elapsed * (self.total_num_lines - self.cur_num_lines) / (200000 * 3600)
            print(f"{self.cur_num_lines}/{self.total_num_lines} lines written in {time_elapsed:.2f}s. "
                  f"Estimated time to completion is {estimated_time:.2f} hours.")
            self.start_time = time.time()

    def close(self):
        for v in self.output_tables.values():
            v.close()


def write_data(path: Path, batch_size: int, total_num_lines: int, outout_queue: Queue):
    writer = Writer(path, batch_size, total_num_lines)
    while True:
        json_object = outout_queue.get()
        if json_object is None:
            break
        writer.write(json_object)
    writer.close()
