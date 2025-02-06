import os
import argparse
import pandas as pd
import numpy as np
import llmabba.llmabba
from llmabba.llmabba import LLMABBA
from sklearn.model_selection import train_test_split
##  Loading models

if __name__=='__main__':

    project_name = "PTBDB"
    task_tpye = "classification"  # classification, regression or forecasting
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    prompt_input = f"""This is a classification task. Identify the "ECG Abnormality" according to the given "Symbolic Series"."""

    abnormal_df = pd.read_csv('../test_data/ptbdb_abnormal.csv', header=None)
    normal_df = pd.read_csv('../test_data/ptbdb_normal.csv', header=None)

    abnormal_length = abnormal_df.shape[0]
    normal_length = normal_df.shape[0]

    Y_data = np.concatenate((np.zeros([abnormal_length], dtype=int), np.ones([normal_length], dtype=int)), axis=0)
    X_data = pd.concat([abnormal_df, normal_df]).to_numpy()

    arranged_seq = np.random.randint(len(Y_data), size=len(Y_data))
    train_data_split = {
        'X_data':0,
        'Y_data':0,
    }

    train_data, test_data, train_target, test_target = train_test_split(
        X_data[arranged_seq, :], Y_data[arranged_seq], test_size=0.2)

    train_data_split['X_data'] = train_data[:500, :]
    train_data_split['Y_data'] = train_target[:500]


    #### Train the LLM models with LoRA
    LLMABBA_classification = LLMABBA()
    model_input, model_tokenizer = LLMABBA_classification.model(
        model_name=model_name,
        max_len=2048
    )
    tokenized_train_dataset, tokenized_val_dataset = LLMABBA_classification.process(
        project_name=project_name,
        data=train_data_split,
        task=task_tpye,
        prompt=prompt_input,
        alphabet_set=-1,
        model_tokenizer=model_tokenizer,
        scalar="z-score",
    )
    LLMABBA_classification.train(
        model_input=model_input,
        num_epochs=1,
        output_dir='../save/',
        train_dataset=tokenized_train_dataset,
        val_dataset=tokenized_val_dataset
    )


    #### YOU CAN *Directly* do the inference with LLM and ABBA, if you have finished the training process
    test_data = np.expand_dims(test_data[1, :], axis=0)
    peft_model_input, model_tokenizer = LLMABBA_classification.model(
        peft_file='/home/kangchen/llm-abba-master/save/checkpoint-25/',
        model_name=model_name,
        max_len=2048)

    out_text = LLMABBA_classification.inference(
        project_name=project_name,
        data=test_data,
        task=task_tpye,
        prompt=prompt_input,
        ft_model=peft_model_input,
        model_tokenizer=model_tokenizer,
        scalar="z-score",
        llm_max_length=256,
        llm_repetition_penalty=1.9,
        llm_temperature=0.0,
        llm_max_new_tokens=2,
    )

    print(out_text)

