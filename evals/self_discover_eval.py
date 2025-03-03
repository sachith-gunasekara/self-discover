import os
import json
import re
import time
import random

from datasets import load_dataset, get_dataset_config_names
from pyprojroot import here
from tqdm import tqdm
import fire

from self_discover import self_discover
from self_discover.helpers.logger import logger
from helpers.llm import model
from helpers.config import config
from helpers.dataset import load_checkpoints
from helpers.eval import calculate_accuracy


def call_self_discover(
    batch,
    phase: int,
    stream: bool,
    reasoning_structure: str = "",
    answer_formats: str = "",
):
    out = self_discover(
        batch["self_discover_input"],
        model,
        reasoning_structure,
        answer_formats,
        phase,
        stream,
    )

    delete_keys = ["task_description", "answer_formats", "task_examples"]

    for key in delete_keys:
        if key in out.keys():
            del out[key]

    if phase == 1:
        logger.debug(out.keys())
        return out
    elif phase == 2:
        batch = batch.map(
            lambda x, idx: {col: out[col][idx] for col in out.keys()}, with_indices=True
        )
        return batch


def format_input_prompt(instance, benchmark):
    if benchmark == "t4d":
        task_description = f"""Observation:
{instance["story"]}

Question:
{instance["question"]}"""

        return {"self_discover_input": task_description}

    if benchmark == "bbh":
        return {"self_discover_input": instance["input"]}

    if benchmark == "math":
        level = instance["level"]
        type = instance["type"]

        one_shot_example = random.sample(
            list(
                load_dataset("qwedsacf/competition_math", split="train").filter(
                    lambda x: x["level"] == level and x["type"] == type
                )
            ),
            1,
        )[0]

        return {
            "self_discover_input": f"""Problem: {instance["problem"]}

<<<BEGIN: An example problem and solution>>>
Problem: {one_shot_example["problem"]}
Solution: {one_shot_example["solution"]}
<<<END: An example problem and solution>>>"""
        }


def get_answer_formats(benchmark):
    if benchmark == "bbh":
        return """- If the answer is not multiple choice, [answer] should be the decided answer. (For eg: Q: not True or False. A: False)
- If the answer is multiple choice,
    - and the given choices are unlabelled options, [answer] should be the chosen option (For eg: Q: Where does the sun rise from? Options: - East, - West, - North. A: East)
    - and the given choices are labelled options, [answer] should be the letter corresponding to the chosen option (For eg: Q: Where does the sun rise from? Options: - A. West, - B. East, - C. North. A: B)"""

    if benchmark == "t4d":
        return """- should be complete with the letter and correct answer from the list of given choices (Example answer:  K. Ananda))"""

    if benchmark == "math":
        return """- should be the final answer based on calculations formatted in Latex style"""


def structure_response(instance):
    text_patterns = ["The final answer is ", "The final answer is: "]
    response = instance["reasoning"]

    for text in text_patterns:
        pattern = rf"(?<={text}).*"
        match = re.search(pattern, response)
        if match:
            answer = match.group(0).strip()
            trajectory = re.sub(pattern, "", response).replace(text, "").strip()

            break
    else:
        answer, trajectory = None, response

    return {"trajectory": trajectory, "answer_pred": answer}


def phaseI(benchmark: str, dataset_name: str, subset: str, phase: int, stream: bool):
    logger.info(
        "Loading subset {} from dataset {}",
        subset,
        dataset_name,
    )

    try:
        dataset = load_dataset(dataset_name, subset, split="train")
    except Exception as e:
        logger.error(e)

        dataset = load_dataset(dataset_name, split="train").filter(
            lambda x: x["type"] == subset
        )

    save_path = here(
        os.path.join(
            "evals/logs",
            config["MODEL"]["model_type"],
            "phaseI",
            benchmark,
        )
    )

    save_file_txt = os.path.join(save_path, f"{benchmark}-{subset}.txt")
    save_file_json = os.path.join(save_path, f"{benchmark}-{subset}.json")

    if os.path.exists(save_file_txt):
        logger.info("Phase I Reasoning Structures already saved in {}", save_file_txt)
        return "skipped"

    os.makedirs(save_path, exist_ok=True)

    dataset = dataset.map(lambda x: format_input_prompt(x, benchmark))

    result = call_self_discover(dataset, phase, stream)

    logger.debug(result.keys())

    with open(save_file_txt, "w") as f:
        f.write(result["reasoning_structure"])
    with open(save_file_json, "w") as f:
        json.dump(result, f, indent=4)

    logger.info("Saving Phase I Reasoning Structures in {}", save_path)


def phaseII(
    benchmark: str, y: str, dataset_name: str, subset: str, phase: int, stream: bool
):
    batch_size = int(config["EVAL"]["batch_size"])

    logger.info(
        "Loading subset {} from dataset {}",
        subset,
        dataset_name,
    )

    try:
        dataset = load_dataset(dataset_name, subset, split="train")
    except Exception as e:
        logger.error(e)

        dataset = load_dataset(dataset_name, split="train").filter(
            lambda x: x["type"] == subset
        )

    save_dir = here(
        os.path.join(
            "evals/logs",
            config["MODEL"]["model_type"],
            "phaseII",
            benchmark,
            f"{benchmark}-{subset}",
        )
    )
    accuracy_file = os.path.join(save_dir, "accuracy.txt")

    if os.path.exists(accuracy_file):
        logger.warning(
            "Evaluation for dataset {}, subset {} has been completed.",
            dataset_name,
            subset,
        )
        return "skipped"

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

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

        checkpoint_path = os.path.join(save_dir, f"checkpoint_{start_idx}_{end_idx}")

        # Skip the batch if it has already been processed
        if os.path.exists(checkpoint_path):
            logger.warning(
                "Batch {}-{} of the dataset {}, subset has already been processed. Proceeding to the next batch.",
                start_idx,
                end_idx,
                dataset_name,
                subset,
            )
            continue

        logger.info("Selecting batch {}-{} for processing.", start_idx, end_idx)
        batch = dataset.select(range(start_idx, end_idx))

        logger.info(
            "Running Phase II on batch {}-{} of the dataset {}, subset {}.",
            start_idx,
            end_idx,
            dataset_name,
            subset,
        )

        with open(
            here(
                os.path.join(
                    "evals/logs",
                    config["MODEL"]["model_type"],
                    "phaseI",
                    benchmark,
                    f"{benchmark}-{subset}.txt",
                )
            ),
            "r",
        ) as f:
            reasoning_structure = f.read()

        result = call_self_discover(
            batch, phase, stream, reasoning_structure, get_answer_formats(benchmark)
        )

        logger.info(
            "Finished processing batch {}-{} of the dataset {}, subset().",
            start_idx,
            end_idx,
            dataset_name,
            subset,
        )
        result.save_to_disk(checkpoint_path)

    logger.info("All batches processed. Loading checkpoints...")

    # Load all checkpoints and concatenate them
    full_dataset = load_checkpoints(save_dir)

    full_dataset = full_dataset.map(structure_response, num_proc=4)

    logger.info("Saving concatenated dataset.")
    full_dataset.save_to_disk(
        here(
            os.path.join(save_dir, f"{benchmark}{f'-{subset}' if subset else ''}_eval")
        )
    )

    logger.info("Concatenated dataset contains {} instances.", len(full_dataset))

    logger.info("Calculating accuracy")

    accuracy = calculate_accuracy(
        benchmark,
        full_dataset[y],
        full_dataset["answer_pred"],
        log_file_path=os.path.join(save_dir, f"{benchmark}_different.txt"),
    )

    logger.info("Accuracy of {} - {}: {}", benchmark, subset, accuracy)

    # Log accuracy
    with open(accuracy_file, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")

    return accuracy


def main(phase: int = -1, stream: bool = True):

    benchmarks = ["t4d", "bbh", "math"]
    y_s = ["answer", "target", "solution"]
    dataset_names = [
        "sachithgunasekara/t4d",
        "maveriq/bigbenchhard",
        "sachithgunasekara/self-discover-MATH-subsample",
    ]
    subset_list = [
        [""],
        get_dataset_config_names(dataset_names[1]),
        list(set(load_dataset(dataset_names[2], split="train")["type"])),
    ]

    for benchmark, y, dataset_name, subsets in zip(
        benchmarks, y_s, dataset_names, subset_list
    ):
        logger.info(
            "Running Self-Discover{} on {}.",
            f" Phase {phase}" if phase in [1, 2] else "",
            benchmark,
        )

        for subset in subsets:

            while True:
                try:
                    if phase == 1:
                        phaseI(benchmark, dataset_name, subset, phase, stream)
                        break
                    elif phase == 2:
                        phaseII(benchmark, y, dataset_name, subset, phase, stream)
                        break

                except Exception as e:
                    # Check for the specific Bearer token error
                    if "Rate limit exceeded" in str(e):
                        wait_time = 10
                        logger.error(
                            f"Rate limit exceeded. Waiting for {wait_time} minutes."
                        )
                        time.sleep(wait_time * 60)
                    else:
                        raise e


if __name__ == "__main__":
    fire.Fire(main)
