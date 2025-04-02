import re
import random

from datasets import load_dataset

from evals.prompts import answer_formats, task_description


def format_input_prompt(
    instance, benchmark: str, few_shot_examples: int = 0, dataset = None
) -> dict:
    if benchmark == "t4d":
        if few_shot_examples != 0:
            few_shot_examples_str = f"\n{'-' * 20}\n".join(
                [
                    task_description.T4D_TASK_DESCRIPTION.format(
                        story=fs_instance["story"], question=fs_instance["question"]
                    )
                    for fs_instance in get_few_shot_instances(
                        dataset, few_shot_examples
                    )
                ]
            )
        result = {
            "self_discover_input": task_description.T4D_TASK_DESCRIPTION.format(
                story=instance["story"], question=instance["question"]
            )
        }

    if benchmark == "bbh":
        if few_shot_examples != 0:
            few_shot_examples_str = "\n\n".join(
                [
                    task_description.BBH_TASK_DESCRIPTION.format(
                        input=instance["input"]
                    )
                    for fs_instance in get_few_shot_instances(
                        dataset, few_shot_examples
                    )
                ]
            )
        result = {
            "self_discover_input": task_description.BBH_TASK_DESCRIPTION.format(
                input=instance["input"]
            ),
        }

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

        result = {
            "self_discover_input": task_description.MATH_TASK_DESCRIPTION.format(
                problem=instance["problem"],
                one_shot_example_problem=one_shot_example["problem"],
                one_shot_example_solution=one_shot_example["solution"],
            )
        }

    print(few_shot_examples)
    if few_shot_examples != 0:
        result["few_shot_examples"] = few_shot_examples_str

    return result


def get_answer_formats(benchmark):
    if benchmark == "bbh":
        return answer_formats.BBH_ANSWER_FORMATS
    if benchmark == "t4d":
        return answer_formats.T4D_ANSWER_FORMATS
    if benchmark == "math":
        return answer_formats.MATH_ANSWER_FORMATS

    return ""


def get_few_shot_instances(dataset, few_shot_examples):
    few_shot_examples_list = random.sample(list(dataset), few_shot_examples)

    return few_shot_examples_list


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
