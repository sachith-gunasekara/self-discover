import os
import sys
import time

from datasets import load_dataset, get_dataset_config_names
from pyprojroot import here
import fire

from tqdm import tqdm

from self_discover import phased_self_discover
from self_discover._helpers.logger import logger

sys.path.append(str(here()))

from evals.helpers import (
    format_input_prompt,
    structure_response,
    get_answer_formats,
    get_few_shot_instances,
)
from evals.helpers.llm import model
from evals.helpers.config import config
from evals.helpers.dataset import load_checkpoints
from evals.helpers.eval import calculate_accuracy


BASE_PATH = here(
    os.path.join("evals/logs/phased_self_discover", config["MODEL"]["model_type"])
)


def call_phased_self_discover(instance, benchmark: str, structured: bool, stream: bool):
    out = phased_self_discover(
        instance["self_discover_input"],
        model,
        get_answer_formats(benchmark),
        structured,
        instance["few_shot_examples"] if "few_shot_examples" in instance.keys() else "",
        stream,
    )

    delete_keys = [
        "task_desctiption",
        "answer_formats",
        "few_shot_examples",
        "task_description_backup",
    ]

    for key in delete_keys:
        if key in out.keys():
            del out[key]

    return out


def evaluate(
    benchmark: str,
    y: str,
    dataset_name: str,
    subset: str,
    structured: bool,
    few_shot_examples: int,
    stream: bool,
):
    batch_size = int(config["EVAL"]["batch_size"])
    save_path = os.path.join(
        BASE_PATH,
        "structured" if structured else "unstructured",
        f"few_shot_{few_shot_examples}",
        benchmark,
        f"{benchmark}-{subset}" if subset else "",
    )
    accuracy_file = os.path.join(save_path, "accuracy.txt")
    full_dataset_file = here(
        os.path.join(save_path, f"{benchmark}{f'-{subset}' if subset else ''}_eval")
    )

    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(accuracy_file) or os.path.exists(full_dataset_file):
        logger.warning(
            "Evaluation for dataset {}, subset {} has been completed",
            dataset_name,
            subset,
        )
        return

    logger.info(
        "Loading subset {} from dataset {}",
        subset,
        dataset_name,
    )
    dataset = load_dataset(dataset_name, subset, split="train")

    logger.info("Formatting input prompts")
    dataset = dataset.map(
        lambda x: format_input_prompt(x, dataset, benchmark, few_shot_examples),
        load_from_cache_file=False,
    )

    logger.info(
        "Generating few shot instances for benchmark {}, subset {}", benchmark, subset
    )

    logger.info(
        "Running evaluations on {} dataset, subset {} in bursts of {}",
        benchmark,
        subset,
        batch_size,
    )

    # Iterate over the dataset in bursts of batch_size
    for start_idx in tqdm(
        range(0, len(dataset), batch_size), desc="Processing batches"
    ):
        end_idx = min(start_idx + batch_size, len(dataset))

        checkpoint_path = os.path.join(save_path, f"checkpoint_{start_idx}_{end_idx}")

        # Skip the batch if it has already been processed
        if os.path.exists(checkpoint_path):
            logger.warning(
                "Batch {}-{} of the dataset {}, subset has already been processed. Proceeding to the next batch",
                start_idx,
                end_idx,
                dataset_name,
                subset,
            )
            continue

        logger.info("Selecting batch {}-{} for processing", start_idx, end_idx)
        batch = dataset.select(range(start_idx, end_idx))

        logger.info(
            "Running Phased Self-Discover on batch {}-{} of the dataset {}, subset {}",
            start_idx,
            end_idx,
            dataset_name,
            subset,
        )

        result = batch.map(
            lambda instance: call_phased_self_discover(
                instance, benchmark, structured, stream
            )
        )

        logger.info(
            "Finished processing batch {}-{} of the dataset {}, {}",
            start_idx,
            end_idx,
            dataset_name,
            subset,
        )
        result.save_to_disk(checkpoint_path)

    logger.info("All batches processed. Loading and concatenating checkpoints...")
    full_dataset = load_checkpoints(save_path)

    logger.info("Extracting final answer from response")
    full_dataset = full_dataset.map(structure_response, num_proc=4)

    logger.info("Saving concatenated dataset")
    full_dataset.save_to_disk(full_dataset_file)

    logger.info("Concatenated dataset contains {} instances", len(full_dataset))

    if benchmark == "math":
        logger.warning("Skipping accuracy calculation for Math dataset")
        return

    logger.info("Calculating accuracy")
    accuracy = calculate_accuracy(
        benchmark,
        full_dataset[y],
        full_dataset["answer_pred"],
        log_file_path=os.path.join(save_path, f"{benchmark}_different.txt"),
    )

    logger.info("Accuracy of {} - {}: {}", benchmark, subset, accuracy)

    # Log accuracy
    with open(accuracy_file, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")

    return accuracy


def main(structured: bool = False, few_shot_examples: int = 0, stream: bool = False):
    benchmarks = ["t4d", "bbh", "math"]
    y_s = ["answer", "target", "solution"]
    dataset_names = [
        "sachithgunasekara/t4d",
        "maveriq/bigbenchhard",
        "sachithgunasekara/self-discover-MATH-subsample",
    ]
    subset_list = [[""], get_dataset_config_names(dataset_names[1]), [""]]

    for benchmark, y, dataset_name, subsets in zip(
        benchmarks, y_s, dataset_names, subset_list
    ):
        logger.info(
            "Running Phased Self-Discover on {}",
            benchmark,
        )

        for subset in subsets:

            while True:
                try:
                    acc = evaluate(
                        benchmark,
                        y,
                        dataset_name,
                        subset,
                        structured,
                        few_shot_examples,
                        stream,
                    )

                    break

                except Exception as e:
                    error_messages = [
                        "Rate limit exceeded",
                        "Requests rate limit exceeded",
                        "Server disconnected without sending a response",
                        "Error response 502",
                        "The read operation timed out",
                        "peer closed connection without sending complete message body"
                    ]
                    if any(msg in str(e) for msg in error_messages):
                        wait_time = config["EVAL"]["wait_time"]

                        logger.error(
                            "Rate limit exceeded. Waiting for {} minutes", wait_time
                        )
                        logger.error(str(e))
                        time.sleep(wait_time * 60)
                    else:
                        raise e


if __name__ == "__main__":
    fire.Fire(main)
