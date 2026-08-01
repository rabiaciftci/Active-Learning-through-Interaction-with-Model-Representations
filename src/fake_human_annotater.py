# simulate_baseline.py
import os
import time
import json
import requests
import pandas as pd

from config import DATA_DIR, HELD_OUT_FOLD, INITIAL_LABELS_PER_CLASS_COUNT, BASE_URL

API         = BASE_URL.rstrip("/")
ANNOTATE_GET = f"{API}/annotate"
HUMAN_POST   = f"{API}/human_annotations"
CSV_PATH = os.path.join(DATA_DIR, "metadata", "UrbanSound8K.csv")

def build_filename_to_code():
    """
    Build a mapping from audio filename -> UrbanSound8K classID.

    We:
      - read UrbanSound8K.csv
      - drop the held-out fold
      - use slice_file_name (e.g. '100032-3-0-0.wav') as key
      - use classID (0..9) as the label code
    """
    df = pd.read_csv(CSV_PATH)

    # keep only training folds (match your model setup)
    train_df = df[df["fold"] != HELD_OUT_FOLD].copy()

    # classID is already 0..9 in UrbanSound8K
    train_df["class_code"] = train_df["classID"].astype(int)

    # build filename -> code dict
    filename2code = {}

    for _, row in train_df.iterrows():
        key = str(row["slice_file_name"])  # e.g. '100032-3-0-0.wav'
        filename2code[key] = int(row["class_code"])

    return filename2code


def get_annotation_items():
    """
    Support response shape:
      {"items":[{"index":123,"filename":"fold1/100032-3-0-0.wav"}, ...]}
    """
    r = requests.get(ANNOTATE_GET, timeout=3)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and "items" in data:
        items = data["items"]
        idxs  = [int(d["index"])    for d in items]
        fns   = [str(d["filename"]) for d in items]
        return idxs, fns

    if isinstance(data, list):
        # If server only returns filenames, we can't use indices consistently.
        raise RuntimeError(
            "Server returned filenames only; please return indices too "
            "(dict with 'items': [{'index':..., 'filename':...}, ...])."
        )

    # Nothing to annotate
    return [], []


def post_human_annotations(indices, filenames, labels):
    payload = {
        "filenames": filenames,
        "indices":   indices,
        "labels":    labels,  # MUST be class codes (ints, 0..9 here)
        "user": "baseline_oracle_from_metadata",
    }
    r = requests.post(HUMAN_POST, json=payload, timeout=3)
    r.raise_for_status()
    return r.json()


def main(poll_every: float = 1.0):
    # Build mapping from *filename* -> true UrbanSound8K classID
    filename2code = build_filename_to_code()
    print(f"[baseline] loaded {len(filename2code)} filename->class mappings")
    print("[baseline] ready; polling for annotation requests...")

    while True:
        try:
            idxs, fns = get_annotation_items()

            if idxs:
                labels = []
                for fn in fns:
                    # In case server uses paths like 'fold3/xxx.wav', strip fold prefix
                    key = os.path.basename(fn)

                    if key not in filename2code:
                        raise KeyError(
                            f"Filename '{fn}' (basename '{key}') not found in metadata; "
                            "check that slice_file_name matches what the API returns."
                        )

                    labels.append(int(filename2code[key]))

                resp = post_human_annotations(idxs, fns, labels)
                print(
                    f"[baseline] labeled {len(idxs)} items "
                    f"| first few: {list(zip(fns[:3], labels[:3]))} "
                    f"| response={resp}"
                )

            time.sleep(poll_every)

        except requests.RequestException as e:
            print(f"[baseline] API error: {e}")
            time.sleep(2.0)
        except Exception as e:
            print(f"[baseline] error: {e}")
            time.sleep(2.0)


if __name__ == "__main__":
    main()
