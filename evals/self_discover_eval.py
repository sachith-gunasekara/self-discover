import os
import time
import random

from datasets import load_dataset, get_dataset_config_names
from pyprojroot import here
from tqdm import tqdm
import fire

from self_discover import self_discover

from helpers.config import config
from self_discover.helpers.logger import logger
from helpers.llm import model


def call_self_discover(
    task_descriptions: list[str], phase: int, stream: bool, answer_formats: str = ""
):
    out = self_discover(task_descriptions, model, answer_formats, phase, stream)

    del out["task_descriptions"]

    return out


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


def bbh(instance):
    answer_formats = """- If the answer is not multiple choice, [answer] should be the decided answer. (For eg: Q: not True or False. A: False)
- If the answer is multiple choice,
    - and the given choices are unlabelled options, [answer] should be the chosen option (For eg: Q: Where does the sun rise from? Options: - East, - West, - North. A: East)
    - and the given choices are labelled options, [answer] should be the letter corresponding to the chosen option (For eg: Q: Where does the sun rise from? Options: - A. West, - B. East, - C. North. A: B)"""

    return call_self_discover(instance["input"], answer_formats)


def t4d(instance):
    task_description = f"""Observation:
{instance["story"]}

Question:
{instance["question"]}"""

    answer_formats = """
- should be complete with the letter and correct answer from the list of given choices (Example answer:  K. Ananda))"""

    return call_self_discover(task_description, answer_formats)


def math(instance):
    level = instance["level"]
    type = instance["type"]

    one_shot_example = random.sample(
        list(
            load_dataset("hendrycks/competition_math", split="train").filter(
                lambda x: x["level"] == level and x["type"] == type
            )
        ),
        1,
    )[0]

    task_description = f"""Problem: {instance["problem"]}

<<<BEGIN: An example problem and solution>>>
Problem: {one_shot_example["problem"]}
Solution: {one_shot_example["solution"]}
<<<END: An example problem and solution>>>"""

    reasoning_formats = """
- should be the final answer based on calculations formatted in Latex style"""

    return call_self_discover(task_description, reasoning_formats)


def evaluate(benchmark: str, y: str, dataset_name: str, subset: str, processor):
    logger.info(
        "Running evaluations on {}. Loading subset {} from dataset {}",
        benchmark,
        subset,
        dataset_name,
    )

    dataset = load_dataset(dataset_name, subset, split="train")

    batch_size = int(config["EVAL"]["batch_size"])

    checkpoint_dir = here(
        os.path.join(
            config["CONFIG"]["checkpoint_dir"], benchmark, f"{benchmark}-{subset}"
        )
    )
    log_dir = here(
        os.path.join(paths.log_dir, "evals", benchmark, f"{benchmark}-{subset}")
    )

    if os.path.exists(os.path.join(checkpoint_dir, f"{benchmark}_eval")):
        logger.debug(
            "The subset %s of the dataset %s has already been processed. Skipping...",
            subset,
            dataset_name,
        )

        return "skipped"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config.add_section("CURRENTS")
    config.set("CURRENTS", "log_dir", str(log_dir))

    logger.info(
        "Running evaluations on %s dataset in bursts of %s", benchmark, batch_size
    )

    # Iterate over the dataset in bursts of batch_size
    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_{start_idx}_{end_idx}"
        )

        # Skip the batch if it has already been processed
        if os.path.exists(checkpoint_path):
            logger.debug(f"Skipping already processed batch {start_idx}-{end_idx}")
            continue

        # Select the batch and run the evaluation
        batch = dataset.select(range(start_idx, end_idx))
        new_ds = batch.map(processor, num_proc=1, load_from_cache_file=False)

        # Save the processed batch to disk as a checkpoint
        new_ds.save_to_disk(checkpoint_path)
        logger.info(f"Saved batch {start_idx}-{end_idx} as checkpoint.")

    logger.info("All batches processed. Loading checkpoints...")

    # Load all checkpoints and concatenate them
    full_dataset = load_checkpoints(checkpoint_dir, benchmark)

    logger.info(f"Combined dataset contains {len(full_dataset)} instances.")

    logger.info("Calculating accuracy")

    accuracy = calculate_accuracy(
        full_dataset,
        benchmark=benchmark,
        y=y,
        y_pred="answer_pred",
        log_file_path=os.path.join(log_dir, f"{benchmark}_different.txt"),
    )

    logger.info("Accuracy of %s - %s: %f", dataset_name, subset, accuracy)

    # Log accuracy
    with open(os.path.join(log_dir, f"{benchmark}.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy}\n")

    return accuracy


def phaseI(benchmark, dataset_name, subset, phase, stream):
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

    save_file = here(
        os.path.join(
            "evals/logs",
            config["MODEL"]["model_type"],
            "phaseI",
            benchmark,
            f"{benchmark}-{subset}.txt",
        )
    )

    if os.path.exists(save_file):
        logger.info("Phase I Reasoning Structures already saved in {}", save_file)
        return "skipped"

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    task_descriptions = dataset.map(lambda x: format_input_prompt(x, benchmark))[
        "self_discover_input"
    ]

    result = call_self_discover(task_descriptions, phase, stream)

    with open(save_file, "w") as f:
        f.write(result["reasoning_structure"])

    logger.info("Saving Phase I Reasoning Structures in {}", save_file)


def phaseII(benchmark, target, dataset_name, subset, processor):

    pass


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

    for benchmark, target, dataset_name, subsets in zip(
        benchmarks, y_s, dataset_names, subset_list
    ):
        logger.info(
            "Running Self-Discover{} on {}.",
            f" Phase {phase}" if phase in [1, 2] else "",
            benchmark,
        )

        for subset in subsets:
            logger.debug("Subset Test {}", subset)

            while True and subset != "":
                try:
                    if phase == 1:
                        phaseI(benchmark, dataset_name, subset, phase, stream)
                        break
                    elif phase == 2:
                        phaseII(benchmark, target, dataset_name, subset, processor)
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
