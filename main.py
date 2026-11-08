
import os
from tqdm import tqdm

from config import *

from src.shuffle_loader import load_stratified_dataset

from src.translator import translate_batch

from src.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    get_completed_indices,
    filter_pending_samples
)

from src.exporter import (
    save_dataset,
    push_to_hub
)

from src.utils import random_delay


def main():

    # ========================================================
    # LOAD STRATIFIED DATASET
    # ========================================================

    print("\nLoading source dataset...")

    dataset = load_stratified_dataset(

        dataset_name=SOURCE_DATASET,

        config_name=CONFIG_NAME,

        split=SPLIT,

        label_column=LABEL_COLUMN,

        start_idx=START_IDX,

        end_idx=END_IDX,

        shuffle_seed=SHUFFLE_SEED,

        cache_dir=DATASET_CACHE_DIR
    )

    if TEXT_COLUMN not in dataset.column_names:

        raise ValueError(
            f"Text column '{TEXT_COLUMN}' not found. "
            f"Available columns: {dataset.column_names}"
        )

    print(
        f"\nSelected samples: {len(dataset)}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    results = load_checkpoint(
        CHECKPOINT_PATH
    )

    completed_indices = get_completed_indices(
        results
    )

    # ========================================================
    # FILTER COMPLETED SAMPLES
    # ========================================================

    pending = filter_pending_samples(
        dataset,
        completed_indices
    )

    if not pending:

        print(
            "\nAll selected samples have already been translated."
        )

    else:

        print(
            f"\nStarting translation of {len(pending)} samples."
        )

        # ====================================================
        # BATCH PROCESSING
        # ====================================================

        request_count = 0

        for i in tqdm(

            range(
                0,
                len(pending),
                BATCH_SIZE
            ),

            desc="Translating batches"

        ):

            batch_records = pending[
                i:i + BATCH_SIZE
            ]

            # ------------------------------------------------
            # Extract texts
            # ------------------------------------------------

            texts = []

            for item in batch_records:

                value = item.get(
                    TEXT_COLUMN
                )

                if value is None:
                    texts.append("")
                else:
                    texts.append(
                        str(value).strip()
                    )

            # ------------------------------------------------
            # Skip empty texts
            # ------------------------------------------------

            valid_positions = [
                idx
                for idx, text in enumerate(texts)
                if text
            ]

            if not valid_positions:

                print(
                    "Skipping batch containing only empty texts."
                )

                continue

            valid_texts = [
                texts[idx]
                for idx in valid_positions
            ]

            valid_records = [
                batch_records[idx]
                for idx in valid_positions
            ]

            # ------------------------------------------------
            # TRANSLATE
            # ------------------------------------------------

            translations = translate_batch(
                valid_texts
            )

            # ------------------------------------------------
            # Validate batch size
            # ------------------------------------------------

            if len(translations) != len(valid_records):

                print(
                    "\nBatch incomplete. "
                    "Nothing saved from this batch."
                )

                continue

            # ------------------------------------------------
            # MERGE ORIGINAL DATA + TRANSLATION
            # ------------------------------------------------

            batch_results = []

            for original, translated in zip(
                valid_records,
                translations
            ):

                result = dict(original)

                result["text_fa"] = translated["text_fa"]

                result["translation_status"] = "completed"

                result["translation_model"] = MODEL

                result["source_language"] = SOURCE_LANGUAGE

                result["target_language"] = TARGET_LANGUAGE

                batch_results.append(
                    result
                )

            # ------------------------------------------------
            # UPDATE CHECKPOINT
            # ------------------------------------------------

            results.extend(
                batch_results
            )

            save_checkpoint(
                results,
                CHECKPOINT_PATH
            )

            request_count += 1

            print(
                f"\nBatch completed: {len(batch_results)} samples"
            )

            print(
                f"Total completed: {len(results)}"
            )

            # ------------------------------------------------
            # REQUEST DELAY
            # ------------------------------------------------

            if USE_RANDOM_DELAY:

                random_delay(
                    REQUEST_DELAY_MIN,
                    REQUEST_DELAY_MAX
                )

            if request_count % REQUESTS_BEFORE_SLEEP == 0:

                print(
                    f"\n{request_count} requests completed. "
                    f"Sleeping for {SLEEP_SECONDS} seconds..."
                )

                import time

                time.sleep(
                    SLEEP_SECONDS
                )

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    save_checkpoint(
        results,
        CHECKPOINT_PATH
    )

    # ========================================================
    # FINAL DATASET
    # ========================================================

    print("\nPreparing final dataset...")

    selected_indices = {
        item["source_index"]
        for item in dataset
    }

    final_results = [
        item
        for item in results
        if item.get("source_index") in selected_indices
    ]

    print(
        f"Translated samples in selected range: {len(final_results)}"
    )

    print(
        f"Selected samples: {len(dataset)}"
    )

    print(
        f"Remaining untranslated: "
        f"{len(dataset) - len(final_results)}"
    )

    # ========================================================
    # SAVE LOCALLY
    # ========================================================

    save_dataset(
        final_results,
        FINAL_DATASET_PATH
    )

    # ========================================================
    # PUSH TO HUGGING FACE
    # ========================================================

    # if final_results:

        # push_to_hub(
        #     final_results,
        #     HF_REPO
        # )

    print("\nTranslation pipeline finished.")


if __name__ == "__main__":

    main()
