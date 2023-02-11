#!/usr/bin/env python3

import glob
import json
import numpy as np
import argparse
from typing import List


openai_creds = None  # {
#  "org": os.environ.get("OPENAI_ORG"),
#  "key": os.environ.get("OPENAI_KEY")
# }


def save_all_stats(all_files: List[str]) -> tuple:
    from formfyxer import lit_explorer

    all_stats = {}
    all_scores = {}
    all_metrics = {}

    # 10MB at N=379
    for idx, f in enumerate(reversed(sorted(all_files))):
        f_name = f.split("/")[-1]
        try:
            all_stats[f_name] = lit_explorer.parse_form(
                f, openai_creds=openai_creds, normalize=True, debug=False
            )
            del all_stats[f_name]["text"]
            del all_stats[f_name]["original_text"]
            if idx % 10 == 0:
                with open("stats.json", "w") as stats_file:
                    stats_file.write(json.dumps(all_stats))
        except Exception as ex:
            print(ex)
            continue
        metrics = lit_explorer._form_complexity_per_metric(all_stats[f_name])
        score = sum(val[2] for val in metrics)
        all_metrics[f_name] = metrics
        all_scores[f_name] = score
        print(f"{f_name}: {all_scores[f_name]}")
        arr = np.array(list(all_scores.values()))
        print(
            f"N: {len(arr)}, New mean: {np.mean(arr)}, new median: {np.median(arr)}, new stddev: {np.std(arr)}"
        )

    for f in all_stats.keys():
        print(f"{f}: {all_scores[f]}")
        print(all_stats[f])
        print()
    return all_stats, all_scores  # , all_metrics


def load_stats() -> tuple:
    from formfyxer.lit_explorer import form_complexity

    with open("stats.json", "r") as stats_file:
        all_stats = json.load(stats_file)
    all_scores = {}
    for f_name in all_stats.keys():
        all_scores[f_name] = form_complexity(all_stats[f_name])
    return all_stats, all_scores


def print_stats(all_stats, all_scores) -> None:
    [
        print(f"{f}: {score}")
        for score, f in sorted([(score, f) for f, score in all_scores.items()])
    ]

    scores_arr = np.array(list(all_scores.values()))
    print(f"Complexity mean: {np.mean(scores_arr)}, stddev: {np.std(scores_arr)}")
    print(
        f"complexity median: {np.median(scores_arr)}, 25% and 75%: {np.percentile(scores_arr, [25, 75])}"
    )

    for key in [
        "time to answer",
        "reading grade level",
        "pages",
        "total fields",
        "avg fields per page",
        "sentences per page",
        "difficult word percent",
        "number of passive voice sentences",
        "passive voice percent",
        "citation count",
    ]:
        arr = np.array([st[key] for st in all_stats.values()])
        print(f"{key} mean: {np.mean(arr)}, stddev: {np.std(arr)}")
        print(f"{key} median: {np.median(arr)}, 25/75%: {np.percentile(arr, [25, 75])}")


def main() -> None:
    """If you pass in a folder with PDFs, we'll calculate stats on all of the PDFs, saving progress
    as we go. If no input folder, we'll try to load saved stats."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in_folder", type=str, help="point to an input folder", default=None
    )
    args = parser.parse_args()
    if args.in_folder != None:
        all_files = glob.glob(
            args.in_folder + ("*" if args.in_folder.endswith("/") else "/*")
        )
        all_stats, all_scores = save_all_stats(all_files)
    else:
        print("Trying to load info from stats.json")
        all_stats, all_scores = load_stats()
    print_stats(all_stats, all_scores)


if __name__ == "__main__":
    main()
