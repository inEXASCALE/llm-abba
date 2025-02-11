import os
import argparse
import pandas as pd
import numpy as np
import llmabba.llmabba
from llmabba.llmabba import LLMABBA
from sklearn.model_selection import train_test_split
##  Loading models

if __name__=='__main__':
    project_name = "ETTh1"
    task_tpye = "forecasting"  # classification, regression or forecasting
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    prompt_input = f"""This is a forecasting task. Forecasting the "Results" according to the given "Symbolic Series"."""
    seq_len_pre = 24  # 96 -> 96;;; 168 -> 24, 48, 96
    seq_len_post = 24  # 96 -> 96;;; 168 -> 24, 48, 96
    MAX_LENGTH_pre = seq_len_pre * 7
    MAX_LENGTH_post = seq_len_post * 7
    MAX_LENGTH = max(MAX_LENGTH_pre, MAX_LENGTH_post)


    df_raw = pd.read_csv('../test_data/ETTh1.csv')
    border1s = [0, 12 * 30 * 24 - seq_len_pre, 12 * 30 * 24 + 4 * 30 * 24 - seq_len_post]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]

    #############################################  Train Data  #############################################
    train_data = df_data[border1s[0]:border2s[0]].values

    #### Train the LLM models with LoRA
    LLMABBA_forecasting = LLMABBA()
    model_input, model_tokenizer = LLMABBA_forecasting.model(
        model_name=model_name,
        max_len=MAX_LENGTH*2
    )
    tokenized_train_dataset, tokenized_val_dataset = LLMABBA_forecasting.process(
        project_name=project_name,
        data=train_data,
        task=task_tpye,
        alphabet_set=-1,
        prompt=prompt_input,
        model_tokenizer=model_tokenizer,
        scalar="z-score",
        seq_len_pre=seq_len_pre,
        seq_len_post=seq_len_post
    )
    LLMABBA_forecasting.train(
        model_input=model_input,
        num_epochs=1,
        output_dir='../save/',
        train_dataset=tokenized_train_dataset,
        val_dataset=tokenized_val_dataset
    )



    #### YOU CAN *Directly* do the inference with LLM and ABBA, if you have finished the training process
    #############################################  Test Data  #############################################
    test_data = df_data[border2s[0]:border2s[2]].values

    test_Xdata = test_data[0:24,:]
    test_Ydata = test_data[25:25+seq_len_post,:]

    if len(test_Xdata.shape) == 1:
        test_Xdata = np.expand_dims(test_Xdata, axis=0)
        test_Ydata = np.expand_dims(test_Ydata, axis=0)

    LLMABBA_forecasting = LLMABBA()
    peft_model_input, model_tokenizer = LLMABBA_forecasting.model(
        peft_file='/home/kangchen/llm-abba-master/save/checkpoint-537/',
        model_name=model_name,
        max_len=4096,
    )

    out_text = LLMABBA_forecasting.inference(
        project_name=project_name,
        data=test_Xdata,
        task=task_tpye,
        prompt=prompt_input,
        ft_model=peft_model_input,
        model_tokenizer=model_tokenizer,
        scalar="z-score",
        llm_max_length=MAX_LENGTH,
        llm_repetition_penalty=1.9,
        llm_temperature=0.0,
        llm_max_new_tokens=256,
        seq_len_pre=24,
        seq_len_post=24,
    )

    print(out_text)




