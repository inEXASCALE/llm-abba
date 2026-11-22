import argparse
import pandas as pd
import numpy as np
from llmabba.llm_abba import LLMABBA
from sklearn.model_selection import train_test_split
from transformers.trainer_utils import get_last_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.1")
    args = parser.parse_args()

    project_name = "PTBDB"
    task_type = "classification"
    model_name = args.model
    output_dir = "../save/PTBDB/checkpoints"
    prompt_input = (
        "Classify the ECG represented by the symbolic series. Use 0 for abnormal and 1 for normal. "
        "Reply with exactly one digit, 0 or 1."
    )

    abnormal_df = pd.read_csv('../test_data/ptbdb_abnormal.csv', header=None)
    normal_df = pd.read_csv('../test_data/ptbdb_normal.csv', header=None)

    abnormal_length = abnormal_df.shape[0]
    normal_length = normal_df.shape[0]

    Y_data = np.concatenate((np.zeros([abnormal_length], dtype=int), np.ones([normal_length], dtype=int)), axis=0)
    X_data = pd.concat([abnormal_df, normal_df], ignore_index=True).iloc[:, :-1].to_numpy()

    train_data, test_data, train_target, test_target = train_test_split(
        X_data, Y_data, test_size=0.2, stratify=Y_data, random_state=42,
    )

    train_data, _, train_target, _ = train_test_split(
        train_data, train_target, train_size=500, stratify=train_target, random_state=42,
    )
    train_data_split = {"X_data": train_data, "Y_data": train_target}


    #### Train the LLM models with LoRA
    LLMABBA_classification = LLMABBA(abba_alpha=0.2)
    model_input, model_tokenizer = LLMABBA_classification.model(
        model_name=model_name,
        max_len=424,
    )
    tokenized_train_dataset, tokenized_val_dataset = LLMABBA_classification.process(
        project_name=project_name,
        data=train_data_split,
        task=task_type,
        prompt=prompt_input,
        alphabet_set=-1,
        model_tokenizer=model_tokenizer,
        scalar="z-score",
        draw_ts = True,
    )
    LLMABBA_classification.train(
        model_input=model_input,
        num_epochs=1,
        output_dir=output_dir,
        train_dataset=tokenized_train_dataset,
        val_dataset=tokenized_val_dataset,
        balance_classes=True,
    )


    # Reload the trained adapter for inference with the saved ABBA codebook.
    test_data = np.expand_dims(test_data[1, :], axis=0)
    peft_model_input, model_tokenizer = LLMABBA_classification.model(
        peft_file=get_last_checkpoint(output_dir),
        model_name=model_name,
        max_len=424,
    )

    out_text = LLMABBA_classification.inference(
        project_name=project_name,
        data=test_data,
        task=task_type,
        prompt=prompt_input,
        ft_model=peft_model_input,
        model_tokenizer=model_tokenizer,
    )

    print(out_text)

