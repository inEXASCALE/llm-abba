"""Train or evaluate one documented LLM-ABBA workflow."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from llmabba import LLMABBA
from llmabba.utils.data_loader import load_from_tsfile_to_dataframe
from llmabba.utils.regressor_tools import calculate_regression_metrics, process_data

DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"
MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
MAX_LEN = 424
HISTORY = 24
HORIZON = 24

PROMPTS = {
    "classification": (
        "Classify the ECG represented by the symbolic series. Use 0 for abnormal and 1 "
        "for normal. Reply with exactly one digit, 0 or 1."
    ),
    "regression": "Estimate heart rate in beats per minute. Reply with one number.",
    "forecasting": "Forecast the next window. Reply with a space-separated sequence of symbols.",
}


def classification_data():
    abnormal = pd.read_csv(DATA_DIR / "ptbdb_abnormal.csv", header=None)
    normal = pd.read_csv(DATA_DIR / "ptbdb_normal.csv", header=None)
    signals = pd.concat([abnormal, normal], ignore_index=True).iloc[:, :-1].to_numpy()
    labels = np.r_[np.zeros(len(abnormal), dtype=int), np.ones(len(normal), dtype=int)]
    remaining, test, remaining_labels, expected = train_test_split(
        signals, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    development, _, targets, _ = train_test_split(
        remaining, remaining_labels, train_size=500, stratify=remaining_labels, random_state=42,
    )
    return {"X_data": development, "Y_data": targets}, test, expected


def regression_data():
    train, targets = load_from_tsfile_to_dataframe(DATA_DIR / "IEEEPPG_TRAIN.ts")
    test, expected = load_from_tsfile_to_dataframe(DATA_DIR / "IEEEPPG_TEST.ts")
    indices = np.random.default_rng(42).choice(len(train), size=500, replace=False)
    # The tutorial uses a short, fixed signal window to limit prompt size.
    development, _ = process_data(train.iloc[indices], min_len=64)
    test, _ = process_data(test, min_len=64)
    return {"X_data": development, "Y_data": targets[indices].astype(float)}, test, expected.astype(float)


def forecasting_data():
    frame = pd.read_csv(DATA_DIR / "ETTh1.csv", parse_dates=["date"]).sort_values("date")
    values = frame[["HUFL", "OT"]].to_numpy(dtype=float)
    # Hold out the window immediately following the first 14 days.
    boundary = 14 * 24
    history = values[boundary - HISTORY:boundary]
    expected = values[boundary:boundary + HORIZON]
    return values[:boundary], history, expected


DATASETS = {
    "classification": classification_data,
    "regression": regression_data,
    "forecasting": forecasting_data,
}


def train_adapter(engine, model_name, project, task, data, checkpoint_dir):
    model, tokenizer = engine.model(model_name=model_name, max_len=MAX_LEN)
    options = {"seq_len_pre": HISTORY, "seq_len_post": HORIZON} if task == "forecasting" else {}
    train_set, val_set = engine.process(
        project_name=project,
        data=data,
        task=task,
        prompt=PROMPTS[task],
        model_tokenizer=tokenizer,
        alphabet_set=-1,
        **options,
    )
    engine.train(
        model_input=model,
        num_epochs=1,
        output_dir=checkpoint_dir,
        train_dataset=train_set,
        val_dataset=val_set,
        balance_classes=task == "classification",
    )


def predict(engine, model_name, project, task, data, checkpoint_dir):
    checkpoint = get_last_checkpoint(checkpoint_dir) if Path(checkpoint_dir).is_dir() else None
    if checkpoint is None:
        raise FileNotFoundError(f"No adapter checkpoint in {checkpoint_dir}. Run the train stage first.")
    model, tokenizer = engine.model(
        model_name=model_name, peft_file=checkpoint, max_len=MAX_LEN,
    )
    options = {}
    if task == "regression":
        # The numeric continuation begins after the prompt, before any EOS.
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_eos_token=False, model_max_length=MAX_LEN,
        )
        tokenizer.pad_token = tokenizer.eos_token
        options["llm_max_new_tokens"] = 16
    elif task == "forecasting":
        options["llm_max_new_tokens"] = 256
    return engine.inference(
        project_name=project,
        data=data,
        task=task,
        prompt=PROMPTS[task],
        ft_model=model,
        model_tokenizer=tokenizer,
        **options,
    )


def evaluate_classification(expected, predictions, training_targets):
    predictions = np.atleast_1d(np.asarray(predictions, dtype=str))
    expected = expected.astype(str)
    labels = ["0", "1"]
    valid = np.isin(predictions, labels)
    majority_label = str(np.bincount(training_targets).argmax())
    print("Accuracy", accuracy_score(expected, predictions))
    print("Recall [abnormal, normal]", recall_score(
        expected, predictions, labels=labels, average=None, zero_division=0,
    ))
    print("Confusion matrix (valid predictions only)", confusion_matrix(
        expected[valid], predictions[valid], labels=labels,
    ), sep="\n")
    print("Invalid predictions", np.count_nonzero(~valid))
    print("Majority-class baseline", np.mean(expected == majority_label))


def evaluate_regression(expected, outputs, training_targets):
    predictions = []
    for output in np.atleast_1d(outputs):
        try:
            predictions.append(float(output))
        except ValueError:
            predictions.append(np.nan)

    predictions = np.asarray(predictions)
    valid = np.isfinite(predictions)
    print("Valid regression responses", np.count_nonzero(valid), "/", len(expected))
    if valid.any():
        print("Model on valid responses")
        print(calculate_regression_metrics(expected[valid], predictions[valid]))
        baseline = np.full(np.count_nonzero(valid), training_targets.mean())
        print("Development-mean baseline on the same rows")
        print(calculate_regression_metrics(expected[valid], baseline))


def evaluate_forecasting(expected, forecast, history):
    persistence = np.broadcast_to(history[-1], expected.shape)
    print("Forecast shape", forecast.shape)
    print("MAE [HUFL, OT]", np.mean(np.abs(expected - forecast), axis=0))
    print("Persistence MAE [HUFL, OT]", np.mean(np.abs(expected - persistence), axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=DATASETS)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--model", default=MODEL, help="Hugging Face model ID or a local model directory")
    parser.add_argument("--no-quantization", action="store_true", help="Load the model without NF4")
    parser.add_argument("--eval-limit", type=int, help="Use the first N held-out rows for a supervised smoke run")
    args = parser.parse_args()
    if args.eval_limit is not None and (
        args.eval_limit < 1 or args.stage != "evaluate" or args.task == "forecasting"
    ):
        parser.error("--eval-limit requires a positive count and supervised evaluation")

    set_seed(42)
    project = f"{args.task}-tutorial"
    checkpoint_dir = str(Path("../save") / project / "checkpoints")
    engine = LLMABBA(abba_alpha=0.2, quant_process=not args.no_quantization)
    development, held_out, expected = DATASETS[args.task]()
    if args.stage == "train":
        train_adapter(engine, args.model, project, args.task, development, checkpoint_dir)
        return

    if args.eval_limit is not None:
        held_out = held_out[:args.eval_limit]
        expected = expected[:args.eval_limit]
    outputs = predict(engine, args.model, project, args.task, held_out, checkpoint_dir)
    if args.task == "classification":
        evaluate_classification(expected, outputs, development["Y_data"])
    elif args.task == "regression":
        evaluate_regression(expected, outputs, development["Y_data"])
    else:
        evaluate_forecasting(expected, outputs, held_out)


if __name__ == "__main__":
    main()
