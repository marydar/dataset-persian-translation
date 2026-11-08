
import os
import json
import tempfile


def load_checkpoint(path):

    if not os.path.exists(path):
        print("No checkpoint found.")
        return []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError(
            "Invalid checkpoint format."
        )

    print(
        f"Loaded {len(results)} completed translations."
    )

    return results


def save_checkpoint(
    results,
    path
):

    directory = os.path.dirname(path) or "."

    os.makedirs(
        directory,
        exist_ok=True
    )

    # Atomic save
    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        suffix=".tmp"
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            temp_path,
            path
        )

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_completed_indices(results):

    return {
        item["source_index"]
        for item in results
        if item.get("source_index") is not None
        and item.get("translation_status") == "completed"
        and item.get("text_fa", "").strip()
    }


def filter_pending_samples(
    dataset,
    completed_indices
):

    pending = []

    for item in dataset:

        source_index = item["source_index"]

        if source_index not in completed_indices:
            pending.append(item)

    print("\nCheckpoint filtering:")
    print(f"Selected samples: {len(dataset)}")
    print(f"Already completed: {len(dataset) - len(pending)}")
    print(f"Remaining samples: {len(pending)}")

    return pending
