from tqdm import tqdm


t4d = (
    lambda y_i, y_pred_i: y_pred_i
    and y_i in y_pred_i
    and y_i == str(y_pred_i.translate(str.maketrans("", "", ".'"))[2:])
)
bbh = lambda y_i, y_pred_i: y_pred_i and y_i.translate(
    str.maketrans("", "", "()")
) == y_pred_i.translate(str.maketrans("", "", '.()"'))


def calculate_accuracy(benchmark, y: list[str], y_pred: list[str], log_file_path: str):
    correct_preds = 0
    for y_i, y_pred_i in tqdm(zip(y, y_pred), desc="Calculating accuracy"):
        if benchmark == "t4d":
            eval_fn = t4d
        elif benchmark == "bbh":
            eval_fn = bbh

        if eval_fn(y_i, y_pred_i):
            correct_preds += 1
        else:
            with open(log_file_path, "a") as f:
                f.write(f"{y_i}, {y_pred_i}\n")
    return correct_preds / len(y_pred)
