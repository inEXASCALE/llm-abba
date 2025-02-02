from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.utils.class_weight import compute_class_weight
from sklearn import preprocessing
import numpy as np
from .abba import XABBA
import pickle
import os
import torch
from datasets.dataset_dict import DatasetDict
from peft import PeftModel
from .utils.fundamentals import *
from .utils.data_loader import load_from_tsfile_to_dataframe
from .utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from .utils.tools import create_directory
from .utils.transformer_tools import fit_transformer


class LLMABBA:
    def __init__(self, abba_tol = 0.000040837, 
                       abba_init = 'agg',
                       abba_alpha = 0.000040837, 
                       bits_for_len = 16,
                       bits_for_inc = 16,
                       abba_scl = 3,
                       abba_verbose = 0, 
                       lora_r = 16,
                       lora_alpha = 16,  
                       target_modules = None,
                        modules_to_save = ["embed_tokens"],
                        lora_dropout = 0.05
        ):  


        self.abba_tol = abba_tol
        self.abba_init = abba_init
        self.abba_alpha = abba_alpha
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.abba_scl = abba_scl
        self.abba_verbose = abba_verbose
        self.lora_r = lora_r 
        self.lora_alpha = lora_alpha
        if target_modules is not None:
            self.target_modules = target_modules
        else:
            self.target_modules = [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                            "lm_head",
                        ]

        self.modules_to_save = modules_to_save
        self.lora_dropout = lora_dropout




    def process(self, data, task, seg_length=16, 
                seq_len_pre = 168, scalar="z-score",
                seq_len_post = 168, data_name="ETTh1"):

        if scalar == "min-max":
            self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        elif scalar == "z-score":
            self.scaler =  preprocessing.StandardScaler()



        if task == "classification":
            pass 
        elif task == "regression":
            pass 
        elif task == "forecasting":
            pass 
        else:
            raise NotImplementedError("Method is not implemented, please contact the maintenance team.")
            


    def build(self, max_len=512, batch_size=4):
        ## Loading Alphabet Set:  you can set vocab_list = pretrained tokens or nothing
        model_tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        model_max_len=max_len,
                        padding_side="right",
                        truncation=True,
                        add_eos_token=True
        )

        model_tokenizer.padding_side = 'right'
        model_tokenizer.pad_token = model_tokenizer.eos_token
        print(len(model_tokenizer))

        mistral_vocab = model_tokenizer.get_vocab()
        vocab_list = list(mistral_vocab.keys())



    def model(self, model_name, max_len):
       
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,  # Mistral, same as before
            quantization_config=bnb_config,  # Same quantization config as before
            device_map="auto",
            trust_remote_code=True,
        )

        model_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_len=max_len,
            padding_side="right",
            truncation=True,
            add_eos_token=True,
        )

        model_tokenizer.padding_side = 'right'
        model_tokenizer.pad_token = model_tokenizer.eos_token
        print(len(model_tokenizer))

        mistral_vocab = model_tokenizer.get_vocab()

        project = "QABBA-" + data_name
        run_name = model_name + "-" + project + "-r-" + str(lora_r)

        output_dir = "./" + run_name
        peft_file = os.listdir(output_dir)
        ft_model = PeftModel.from_pretrained(model, output_dir + "/" + peft_file[0])


    def train(self):
        pass



    def inference(self):
        pass