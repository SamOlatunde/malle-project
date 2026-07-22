"""
Module: temp_eval.py

Computes instance-level and class-level Hit Rate@3 for a given results file.

Hit Rate@3 = fraction of queries for which the correct instance / class
appears at least once among the top-3 matches. Since every instance_id is
unique within the index, there is exactly one relevant item per query at
the instance level, so this is the right metric to report (precision@k
would just be the same 0/1 signal divided by k, with no extra information).

Usage:
    python temp_eval.py path/to/results.jsonl
"""

import argparse

from embed import load_jsonl

K = 3


def compute_hit_rate(results, k=K):
    """Returns {'instance_hit_rate': ..., 'class_hit_rate': ...} at top-k."""
    total = len(results)
    instance_hits = 0
    class_hits = 0

    for r in results:
        true_class = r['query_class']
        true_instance_id = r['query_instance_id']
        topk = r['matches'][:k]

        class_hit = any(m['index_class'] == true_class for m in topk)
        instance_hit = any(
            m['index_class'] == true_class and m['index_instance_id'] == true_instance_id
            for m in topk
        )

        class_hits += class_hit
        instance_hits += instance_hit

    return {
        'instance_hit_rate': instance_hits / total,
        'class_hit_rate': class_hits / total,
    }


def main():
    parser = argparse.ArgumentParser(description=f'Compute Hit Rate@{K} from a results JSONL file.')
    parser.add_argument('results_path', help='Path to a results .jsonl file')
    args = parser.parse_args()

    results = load_jsonl(args.results_path)
    if not results:
        parser.error(f'No records found in {args.results_path}')

    stats = compute_hit_rate(results)

    print(f'[{args.results_path}]  ({len(results)} queries)\n')
    print(f'  Instance Hit Rate@{K} = {stats["instance_hit_rate"]:.4f}')
    print(f'  Class    Hit Rate@{K} = {stats["class_hit_rate"]:.4f}')


if __name__ == '__main__':
    main()