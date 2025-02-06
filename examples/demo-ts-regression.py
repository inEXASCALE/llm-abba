import os
import argparse
import pandas as pd
import numpy as np
import llmabba.llmabba
# from data_loader import load_from_tsfile_to_dataframe
# from regressor_tools import process_data, fit_regressor, calculate_regression_metrics
# from sklearn import preprocessing
from llmabba.llmabba import LLMABBA
from llmabba.utils.data_loader import load_from_tsfile_to_dataframe
from llmabba.utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from sklearn.model_selection import train_test_split
##  Loading models

if __name__=='__main__':

    project_name = "IEEEPPG"
    task_tpye = "regression"  # classification, regression or forecasting
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    prompt_input = f"""This is a regression task. Score the "ECG Abnormality" according to the given "Symbolic Series"."""

    train_file = "../test_data/IEEEPPG_TRAIN.ts"
    test_file = "../test_data/IEEEPPG_TEST.ts"

    X_data, Y_data = load_from_tsfile_to_dataframe(train_file)
    train_data_split = {
        'X_data':0,
        'Y_data':0,
    }

    train_data_split['X_data'] = X_data
    train_data_split['Y_data'] = Y_data

    #### Train the LLM models with LoRA
    LLMABBA_regression = LLMABBA()
    # model_input, model_tokenizer = LLMABBA_regression.model(
    #     model_name=model_name,
    #     max_len=2048
    # )
    # tokenized_train_dataset, tokenized_val_dataset = LLMABBA_regression.process(
    #     project_name=project_name,
    #     data=train_data_split,
    #     task=task_tpye,
    #     prompt=prompt_input,
    #     alphabet_set=-1,
    #     model_tokenizer=model_tokenizer,
    #     scalar="z-score",
    # )
    # LLMABBA_regression.train(
    #     model_input=model_input,
    #     num_epochs=1,
    #     output_dir='../save/' + project_name + '/',
    #     train_dataset=tokenized_train_dataset,
    #     val_dataset=tokenized_val_dataset
    # )


    #### YOU CAN *Directly* do the inference with LLM and ABBA, if you have finished the training process
    X_test, Y_test = load_from_tsfile_to_dataframe(test_file)

    min_len = np.inf
    for i in range(len(X_test)):
        x = X_test.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    # print("[{}] Minimum length: {}".format(module, min_len))

    X_test_processed, _ = process_data(X_test, min_len=min_len)

    test_input = X_test_processed[1, :, :]
    test_target = Y_test[1]


    peft_model_input, model_tokenizer = LLMABBA_regression.model(
        peft_file='/home/kangchen/llm-abba-master/save/IEEEPPG/checkpoint-70/',
        model_name=model_name,
        max_len=2048)

    out_text = LLMABBA_regression.inference(
        project_name=project_name,
        data=test_input,
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

