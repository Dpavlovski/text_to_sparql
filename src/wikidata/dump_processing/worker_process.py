from collections import defaultdict
from multiprocessing import Queue

import ujson


def process_json(obj):
    out_data = defaultdict(list)
    id = obj['id']

    out_data['labels'].append({
        'qid': id,
        'label_en': obj['labels']['en']['value'] if obj['labels'].get('en', {}) else None,
        'label_de': obj['labels']['de']['value'] if obj['labels'].get('de', {}) else None,
        'label_ru': obj['labels']['ru']['value'] if obj['labels'].get('ru', {}) else None,
    })

    out_data['descriptions'].append({
        'qid': id,
        'description_en': obj['descriptions']['en']['value'] if obj['descriptions'].get('en', {}) else None,
        'descriptions_de': obj['descriptions']['de']['value'] if obj['descriptions'].get('de', {}) else None,
        'description_ru': obj['descriptions']['ru']['value'] if obj['descriptions'].get('ru', {}) else None,
    })

    return dict(out_data)


def process_data(work_queue: Queue, output_queue: Queue):
    while True:
        json_obj = work_queue.get()
        if json_obj is None:
            break
        try:
            output_queue.put(process_json(ujson.loads(json_obj)))
        except Exception as e:
            continue
