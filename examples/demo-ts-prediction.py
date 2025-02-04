import os
import argparse
import matplotlib.pyplot as plt
from fABBA import JABBA
import torch
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_provider.data_factory import data_provider

import warnings


torch.cuda.empty_cache()


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)



# def model_preprocessing_function(examples):
#     return model_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

# Python program to convert a list to string
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + ' '
    # return string
    return str1

# Python program to convert a list to string
def stringToList(s):
    # initialize an empty string
    str1 = []
    # traverse in the string
    for ele in range(int(len(s)/2)):
        str1.append(s[ele*2])
    # return string
    return str1



# def generate_and_tokenize_prompt(data_point):

#     full_prompt = f"""Generate a symbolic "Series" based on a given symbolic "Inputs". \n

#         ### Inputs:
#         {data_point["text"]}.

#         ### Series:
#         {data_point["label"]}
#         """
#     # print(full_prompt)
#     return tokenize(full_prompt)


def tokenize(data_point):
    model_inputs = model_tokenizer(
        data_point["text_inputs"],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )

    result = model_tokenizer(
        data_point["text_outputs"],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )

    model_inputs["labels"] = result["input_ids"].copy()

    #     result["labels"] = result["input_ids"].copy()
    #     result["labels_mask"] = result["attention_mask"].copy()

    return model_inputs


def main(opts):

    warnings.filterwarnings("ignore")

    global batch_size
    global model_tokenizer
    global MAX_LENGTH

    batch_size = opts.batch_size
    MAX_LENGTH = opts.MAX_LENGTH

    ##  Quantization Coonfig
    quantization_config = BitsAndBytesConfig(
        # Load the model with 4-bit quantization
        load_in_4bit=True,
        # Use double quantization
        bnb_4bit_use_double_quant=True,
        # Use 4-bit Normal Float for storing the base model weights in GPU memory
        bnb_4bit_quant_type="nf4",
        # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    seq_len = opts.seq_len
    ###############   UCR2018 data   ###############
    if (opts.data_name == 'ETTh1') or (opts.data_name == 'ETTh2'):

        current_file = 'data/time-series-dataset/dataset/ETT-small/'

        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(current_file, opts.data_name+'.csv'))
        border1s = [0, 12 * 30 * 24 - opts.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - opts.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    elif (opts.data_name == 'ETTm1') or (opts.data_name == 'ETTm2'):
        current_file = 'data/time-series-dataset/dataset/ETT-small/'

        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(current_file, opts.data_name+'.csv'))
        border1s = [0, 12 * 30 * 24 * 4 - opts.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - opts.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

    elif (opts.data_name == 'Weather'):
        current_file = 'data/time-series-dataset/dataset/ETT-small/'

        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(current_file, opts.data_name + '.csv'))
        border1s = [0, 6 * 30 * 24 * 6 - opts.seq_len, 6 * 30 * 24 * 6 + 0 * 30 * 24 * 6 - opts.seq_len]
        border2s = [6 * 30 * 24 * 6, 6 * 30 * 24 * 6 + 0 * 30 * 24 * 6, 6 * 30 * 24 * 6 + 6 * 30 * 24 * 6]

    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]

    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(train_data.values)

    data_patch = np.zeros([data.shape[0] - seq_len, seq_len, data.shape[1]], dtype=float)
    for i_data_patch in range(data.shape[0] - seq_len):
        data_patch[i_data_patch, :, :] = data[i_data_patch:i_data_patch + seq_len, :]

    jabba = JABBA(tol=opts.ABBA_tol, init=opts.ABBA_init,
                  alpha=opts.ABBA_alpha, scl=3, verbose=1)
    train_symbols = jabba.fit_transform(data_patch)

    print(len(train_symbols[1]))

    # reconst = jabba.inverse_transform(train_symbols)  # convert into array
    # reconst_same_shape = jabba.recast_shape(reconst)  # recast into original shape
    # print(np.linalg.norm((data_patch - reconst_same_shape).reshape(-1, np.prod(data_patch.shape[1:])), 'fro'))

    symbols_convert = []
    for i_data in range(len(train_symbols)):
        symbols_convert.append(listToString(list(train_symbols[i_data])))

    train_data_symbolic = symbols_convert[:border1s[1] - border1s[0] - seq_len]
    train_target_symbolic = symbols_convert[seq_len:border1s[1] - border1s[0]]

    arranged_seq = np.random.randint(len(train_data_symbolic), size=int(len(train_data_symbolic) * 0.2))

    val_data_symbolic = [train_data_symbolic[index] for index in arranged_seq]
    val_target_symbolic = [train_target_symbolic[index] for index in arranged_seq]

    data_TS = DatasetDict({
        'train': Dataset.from_dict({'text_outputs': train_target_symbolic, 'text_inputs': train_data_symbolic}),
        'val': Dataset.from_dict({'text_outputs': val_target_symbolic, 'text_inputs': val_data_symbolic}),
    })



    ######################################################################## lora config
    if opts.model_name == "roberta-large":

        model_checkpoint = "roberta-large"
        model_tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            truncation=True,
            add_eos_token=True
        )
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token
        model_tokenizer.padding_side = 'right'

        model_input = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            quantization_config=quantization_config,
            device_map="auto"
        )

        from peft import prepare_model_for_kbit_training

        model_input.gradient_checkpointing_enable()
        model_input.config.pad_token_id = model_input.config.eos_token_id
        model_input = prepare_model_for_kbit_training(model_input)

        roberta_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["embed_tokens"],
        )
        model_input = get_peft_model(model_input, roberta_peft_config)

    elif opts.model_name == "mistral-7B":

        model_checkpoint = 'mistralai/Mistral-7B-Instruct-v0.1'
        model_tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            truncation=True,
            add_eos_token=True
        )

        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token
        model_tokenizer.padding_side = 'right'

        model_input = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            quantization_config=quantization_config,
            device_map="auto"
        )

        from peft import prepare_model_for_kbit_training

        model_input.gradient_checkpointing_enable()

        model_input.config.pad_token_id = model_input.config.eos_token_id
        model_input = prepare_model_for_kbit_training(model_input)

        mistral_lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,  # the dimension of the low-rank matrices
            lora_alpha=16,  # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            modules_to_save=["embed_tokens"],
            lora_dropout=0.05,  # dropout probability of the LoRA layers
            bias='none',  # wether to train bias weights, set to 'none' for attention layers
        )
        model_input = get_peft_model(model_input, mistral_lora_config)

    elif opts.model_name == "llama2-7B":

        model_checkpoint = "starmpcc/Asclepius-Llama2-7B"
        model_tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            truncation=True,
            add_eos_token=True
        )
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token

        model_input = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            quantization_config=quantization_config,
            device_map="auto"
        )

        from peft import prepare_model_for_kbit_training

        model_input.gradient_checkpointing_enable()
        model_input.config.pad_token_id = model_input.config.eos_token_id
        model_input = prepare_model_for_kbit_training(model_input)

        llama_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            modules_to_save=["embed_tokens"],
        )
        model_input = get_peft_model(model_input, llama_peft_config)

    else:
        print("Please input correct models!")


    model_input.print_trainable_parameters()
    model_input = model_input.cuda()

    # model_tokenized_datasets = data_TS.map(model_preprocessing_function, batched=True)

    # model_tokenized_datasets = data_TS.map(generate_and_tokenize_prompt)
    model_tokenized_datasets = data_TS.map(tokenize, batched=True, batch_size=batch_size)
    model_tokenized_datasets.set_format("torch")

    project = "ts-finetune-" + opts.data_name
    # b-instruct-v0.1-h
    run_name = opts.model_name + "-" + project + "-r-" + str(opts.lora_r) + "-L-" + str(opts.seq_len)
    output_dir = "./" + run_name

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=opts.epochs,
        optim="paged_adamw_8bit",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,
        save_total_limit=1,
    )

    trainer_abba = Trainer(
        model=model_input,
        args=training_args,
        train_dataset=model_tokenized_datasets['train'],
        eval_dataset=model_tokenized_datasets["val"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=model_tokenizer, mlm=False)
    )

    model_input.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer_abba.train()

    ################################  Evaluate  ################################
    from peft import PeftModel

    peft_folder = os.listdir(output_dir)
    ft_model = PeftModel.from_pretrained(
        model_input,
        output_dir + "/" + peft_folder[0]
    )

    test_data = df_data[border2s[0]:border2s[2]]
    # scaler.fit(train_data.values)
    data = scaler.transform(test_data.values)

    data_patch = np.zeros([data.shape[0], data.shape[1], seq_len], dtype=float)
    for i_data_patch in range(data.shape[0] - seq_len):
        data_patch[i_data_patch, :, :] = np.transpose(data[i_data_patch:i_data_patch + seq_len, :])


    test_symbols, test_params = jabba.transform(data_patch)
    symbols_convert = []
    for i_data in range(len(test_symbols)):
        symbols_convert.append(listToString(list(test_symbols[i_data])))

    test_data_symbolic = symbols_convert[:border2s[2] - border2s[0] - seq_len]
    test_target_symbolic = symbols_convert[seq_len:border2s[2] - border2s[0] + seq_len]

    symbols_LLM = []
    test_length = 100 # len(test_data_symbolic)
    for i_test in range(test_length):  # len(data_TS['test'])):

        print('###################################  Model Outputs  ####################################')
        model_input = model_tokenizer(test_data_symbolic[i_test], return_tensors="pt").to("cuda")

        model_output = model_tokenizer.decode(
            ft_model.generate(
                **model_input,
                max_new_tokens=8,
                max_length=MAX_LENGTH,
                repetition_penalty=2.1
            )[0],
            skip_special_tokens=True
        )

        model_output_list = model_output.split(' ')
        model_output_list_copy = model_output_list.copy()
        for i_remove in range(len(model_output_list_copy)):
            if model_output_list_copy[i_remove] not in jabba.parameters.alphabets:
                model_output_list.remove(model_output_list_copy[i_remove])

        symbols_LLM.append(model_output_list)

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # test_length = 50

    reconst_test = jabba.inverse_transform(symbols_LLM, test_params)  # convert into array

    reconst_same_shape = np.zeros(([test_length, 7, seq_len]), dtype=float)

    Y_pred_all = np.zeros(([test_length, 7 * seq_len]), dtype=float)
    Y_true_all = np.zeros(([test_length, 7 * seq_len]), dtype=float)
    MAE_result = np.zeros(test_length, dtype=float)
    MSE_result = np.zeros(test_length, dtype=float)

    for i_reconst in range(len(reconst_test)):
        try:
            reconst_same_shape[i_reconst, :, :] = np.reshape(reconst_test[i_reconst][:7 * seq_len],
                                                             (7, seq_len))  # recast into original shape
        except:
            print("An exception occurred: " + str(len(reconst_test[i_reconst])))
            continue

        #     Y_pred = scaler.inverse_transform(np.transpose(reconst_same_shape[i_reconst, :, :]))
        Y_pred = scaler.inverse_transform(np.transpose(reconst_same_shape[i_reconst, :, :]))
        Y_pred = np.reshape(Y_pred, (1, 7 * seq_len))
        Y_true = np.reshape(test_data.values[i_reconst + seq_len:i_reconst + seq_len * 2, :], (1, 7 * seq_len))

        Y_pred_all[i_reconst, :] = Y_pred
        Y_true_all[i_reconst, :] = Y_true

        MAE_result[i_reconst] = np.mean(np.abs(Y_pred - Y_true))
        MSE_result[i_reconst] = np.mean(np.power(Y_pred - Y_true, 2))

    print('###############################')
    print('MSE: ')
    print(mean_squared_error(Y_true_all, Y_pred_all))
    print('MAE: ')
    print(mean_absolute_error(Y_true_all, Y_pred_all))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Times Series')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    parser.add_argument('--data_name', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='llama2-7B')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--ABBA_tol', type=float, default=0.05)
    parser.add_argument('--ABBA_alpha', type=float, default=0.05)
    parser.add_argument('--ABBA_init', type=str, default='agg')
    parser.add_argument('--ABBA_k', type=int, default=1000)
    parser.add_argument('--ABBA_scl', type=int, default=3)
    parser.add_argument('--UCR_data_num', type=int, default=1)
    parser.add_argument('--MAX_LENGTH', type=int, default=2048)

    args = parser.parse_args()
    # print('***********************************************************************************')
    main(args)
    # print('***********************************************************************************')




