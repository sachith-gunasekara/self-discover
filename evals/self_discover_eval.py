import os
import sys
import json
import time

from datasets import load_dataset, get_dataset_config_names
from pyprojroot import here
import fire
from tqdm.auto import tqdm

from self_discover import self_discover
from self_discover._helpers.logger import logger
from self_discover._helpers.enums import Phase

sys.path.append(str(here()))

from evals.helpers import format_input_prompt, structure_response, get_answer_formats
from evals.helpers.llm import model
from evals.helpers.config import config
from evals.helpers.dataset import load_checkpoints
from evals.helpers.eval import calculate_accuracy


BASE_PATH = here(
    os.path.join("evals/logs/self_discover", config["MODEL"]["model_type"])
)


def call_self_discover(
    batch,
    answer_formats: str,
    phase: int,
    stream: bool,
    reasoning_structure: str = "",
):
    out = self_discover(
        batch["self_discover_input"],
        model,
        answer_formats,
        reasoning_structure,
        phase,
        stream,
    )        

    delete_keys = ["task_description", "answer_formats", "task_examples"]

    for key in delete_keys:
        if key in out.keys():
            del out[key]

    if phase == Phase.I.value:
        logger.debug(out.keys())
        return out
    elif phase == Phase.II.value:
        batch = batch.map(
            lambda x, idx: {col: out[col][idx] for col in out.keys()}, with_indices=True
        )
        return batch


def phaseI(benchmark: str, dataset_name: str, subset: str, phase: int, stream: bool):
    logger.info(
        "Loading subset {} from dataset {}",
        subset,
        dataset_name,
    )

    dataset = load_dataset(dataset_name, subset, split="train")

    save_path = os.path.join(
        BASE_PATH,
        "phaseI",
        benchmark,
    )

    os.makedirs(save_path, exist_ok=True)

    save_file_txt = os.path.join(save_path, f"{benchmark}-{subset}.txt")
    save_file_json = os.path.join(save_path, f"{benchmark}-{subset}.json")

    if os.path.exists(save_file_txt):
        logger.info("Phase I Reasoning Structures already saved in {}", save_file_txt)
        return

    dataset = dataset.map(lambda x: format_input_prompt(x, benchmark))

    result = call_self_discover(dataset, "", phase, stream)

    logger.debug(result.keys())

    with open(save_file_txt, "w") as f:
        f.write(result["reasoning_structure"])
    with open(save_file_json, "w") as f:
        json.dump(result, f, indent=4)

    logger.info("Saving Phase I Reasoning Structures in {}", save_path)

    return result["reasoning_structure"]


def phaseII(
    benchmark: str, y: str, dataset_name: str, subset: str, phase: int, stream: bool
):
    batch_size = int(config["EVAL"]["batch_size"])
    save_path = os.path.join(
        BASE_PATH,
        "phaseII",
        benchmark,
        f"{benchmark}-{subset}" if subset else "",
    )
    accuracy_file = os.path.join(save_path, "accuracy.txt")
    full_dataset_file = here(
        os.path.join(save_path, f"{benchmark}{f'-{subset}' if subset else ''}_eval")
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
    dataset = dataset.map(lambda x: format_input_prompt(x, benchmark))

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
            "Running Phase II on batch {}-{} of the dataset {}, subset {}",
            start_idx,
            end_idx,
            dataset_name,
            subset,
        )

        with open(
            here(
                os.path.join(
                    BASE_PATH,
                    "phaseI",
                    benchmark,
                    f"{benchmark}-{subset}.txt",
                )
            ),
            "r",
        ) as f:
            reasoning_structure = f.read()

        result = call_self_discover(
            batch, get_answer_formats(benchmark), phase, stream, reasoning_structure
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


def main(phase: int = Phase.BOTH.value, stream: bool = False):

    benchmarks = ["t4d", "bbh"]
    y_s = ["answer", "target"]
    dataset_names = ["sachithgunasekara/t4d", "maveriq/bigbenchhard"]
    subset_list = [[""], get_dataset_config_names(dataset_names[1])]

    for benchmark, y, dataset_name, subsets in zip(
        benchmarks, y_s, dataset_names, subset_list
    ):
        for subset in subsets:
            if subset == "dyck_languages":
                logger.warning("Skipping {} of {}", subset, benchmark)
                continue
            
            logger.info(
                "Running Self-Discover{} on benchmark {}{}",
                f" Phase {phase}" if phase in [Phase.I.value, Phase.II.value] else "",
                benchmark,
                f"{f', subset {subset}' if subset else ''}",
            )
            while True:
                try:
                    if phase == Phase.I.value:
                        phaseI(benchmark, dataset_name, subset, phase, stream)
                        break
                    elif phase == Phase.II.value:
                        phaseII(benchmark, y, dataset_name, subset, phase, stream)
                        break

                except Exception as e:
                    error_messages = [
                        # "Rate limit exceeded",
                        # "Requests rate limit exceeded",
                        # "Server disconnected without sending a response",
                        # "Error response 502",
                        # "The read operation timed out",
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