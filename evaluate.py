"""
Module: evaluate.py

Computes instance-level and class-level Recall@k from search results.
All paths are derived from config.py — change knobs there, not here.
"""

import config
from embed import load_jsonl


results = load_jsonl(config.RESULTS)

k_list = [1, 3, 5, 10, 11]

for k in k_list:
    tp_instance = 0
    tp_class    = 0
    total       = 0

    for r in results:
        total += 1
        true_class       = r['query_class']
        true_instance_id = r['query_instance_id']

        topk_classes   = [m['index_class']       for m in r['matches'][:k]]
        topk_instances = [m['index_instance_id'] for m in r['matches'][:k]]

        # A class match is counted once even if it appears multiple times in
        # top-k. An instance is only counted correct when it also shares the
        # right class (prevents false positives from same instance_id across
        # different classes).
        counted_class = False
        for cls, inst in zip(topk_classes, topk_instances):
            if true_class == cls:
                if not counted_class:
                    tp_class   += 1
                    counted_class = True
                if true_instance_id == inst:
                    tp_instance += 1

    print(f'[{config.RUN_TAG} | variant={config.DATASET_VARIANT}]')
    print(f'  Instance Recall@{k} = {tp_instance / total:.4f}')
    print(f'  Class    Recall@{k} = {tp_class    / total:.4f}', end='\n\n')
