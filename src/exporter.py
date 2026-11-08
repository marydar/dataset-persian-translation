
import os
import shutil

from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets
)


def prepare_for_export(results):

    cleaned = []

    for item in results:

        if not isinstance(item, dict):
            continue

        if item.get("translation_status") != "completed":
            continue

        text = item.get("text")
        text_fa = item.get("text_fa")

        if not isinstance(text, str):
            continue

        if not isinstance(text_fa, str):
            continue

        if not text.strip() or not text_fa.strip():
            continue

        cleaned.append(item)

    return cleaned


def save_dataset(
    results,
    path
):

    results = prepare_for_export(
        results
    )

    dataset = Dataset.from_list(
        results
    )

    # Remove previous export if it exists
    if os.path.exists(path):

        shutil.rmtree(path)

    dataset.save_to_disk(
        path
    )

    print(
        f"Saved {len(dataset)} translated samples to {path}"
    )

    return dataset


def push_to_hub(
    results,
    repo_id
):

    results = prepare_for_export(
        results
    )

    if not results:
        print("No translated samples to upload.")
        return None

    new_dataset = Dataset.from_list(
        results
    )

    try:

        print(
            "Downloading existing Hugging Face dataset..."
        )

        old_dataset = load_dataset(
            repo_id,
            split="train"
        )

        print(
            f"Existing samples: {len(old_dataset)}"
        )

        # Ensure matching column order
        old_columns = set(old_dataset.column_names)
        new_columns = set(new_dataset.column_names)

        if old_columns != new_columns:

            raise ValueError(
                "Existing and new dataset schemas differ. "
                f"Existing: {old_dataset.column_names}; "
                f"New: {new_dataset.column_names}"
            )

        new_dataset = new_dataset.select_columns(
            old_dataset.column_names
        )

        merged = concatenate_datasets(
            [
                old_dataset,
                new_dataset
            ]
        )

    except Exception as e:

        # Only create a new repository dataset when
        # the repository does not exist.
        from huggingface_hub.utils import RepositoryNotFoundError

        if isinstance(e, RepositoryNotFoundError):

            print(
                "Repository does not exist. Creating new dataset."
            )

            merged = new_dataset

        else:
            raise

    # --------------------------------------------------------
    # Remove duplicate source indices
    # --------------------------------------------------------

    if "source_index" in merged.column_names:

        df = merged.to_pandas()

        before = len(df)

        df = df.drop_duplicates(
            subset=["source_index"],
            keep="last"
        ).reset_index(drop=True)

        print(
            f"Removed {before - len(df)} duplicate source indices."
        )

        merged = Dataset.from_pandas(
            df,
            preserve_index=False
        )

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    merged.push_to_hub(
        repo_id
    )

    print(
        f"Uploaded {len(merged)} samples to {repo_id}"
    )

    return merged
