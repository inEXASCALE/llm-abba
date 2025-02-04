from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from accelerate import FullyShardedDataParallelPlugin, Accelerator
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model, TaskType
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
from sklearn.model_selection import train_test_split
from datetime import datetime
from datasets.dataset_dict import DatasetDict
from datasets import Dataset

def save_abba(model, filename):
    pickle.dump(model, file = open(filename, "wb"))

def load_abba(filename):
    return pickle.load(open(filename, "rb"))


def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""This is a forecasting task. Forecasting the "Results" according to the given "Symbolic Series".

    ### Symbolic Series: {data_point["text_inputs"]}

    ### Results: {data_point["text_outputs"]}

    """
    return tokenize(full_prompt)

class LLMABBA:
    def __init__(self,
                 abba_tol = 0.05,
                 abba_init = 'agg',
                 abba_alpha = 0.01,
                 bits_for_len = 16,
                 bits_for_inc = 16,
                 abba_scl = 3,
                 abba_verbose = 0,
                 lora_r = 16,
                 lora_alpha = 16,
                 target_modules = None,
                 modules_to_save = ["embed_tokens"],
                 lora_dropout = 0.05,
                 quant_process = True
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
        self.quant_process = quant_process
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


    def tokenize(self, prompt_input):
        result = self.model_tokenizer(
            prompt_input,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def build(self, data_point):
        """Build the Prompt Data"""
        if self.inference_mode:
            pass
        else:
            full_prompt = f"""{self.prompt}
            ### Symbolic Series: {data_point["text_inputs"]}
            ### Results: {data_point["text_outputs"]}
            """
        return self.tokenize(full_prompt)


    def process(self, data, task, prompt, model_tokenizer,
                seq_len_pre = 24, scalar="z-score",
                seq_len_post = 24, inference_mode=False, alphabet_set=None)-> None:
        """Load data and process data"""

        if alphabet_set is None:
            vocab_list = model_tokenizer.get_vocab()
            alphabet_set = list(vocab_list.keys())
            print("Using LLM tokens as the alphabet_set")
        else:
            print("Using self-defined alphabet_set")

        self.task = task
        self.alphabet_set = alphabet_set
        self.inference_mode = inference_mode
        self.prompt = prompt
        self.model_tokenizer = model_tokenizer

        # Initialize the StandardScaler
        if scalar == "min-max":
            self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        elif scalar == "z-score":
            self.scaler =  preprocessing.StandardScaler()

        # Initialize the ABBA block
        self.xabba = XABBA(tol=self.abba_tol, init=self.abba_init,
                           alpha=self.abba_alpha, scl=self.abba_scl, verbose=0)

        if self.inference_mode is False:
            if self.task == "classification":
                # Fit the scaler to the training data and transform it
                X_data = self.scaler.fit_transform(data['X_data'])
                Y_data = data['Y_data']

                symbols = self.xabba.fit_transform(X_data, alphabet_set=alphabet_set)
                reconstruction = self.xabba.inverse_transform(symbols)
                # reconst_same_shape = self.xabba.recast_shape(reconstruction)  # recast into original shape

                symbols_convert = []
                for i_data in range(len(symbols)):
                    symbols_convert.append(listToString(list(symbols[i_data])))

                train_data_symbolic, val_data_symbolic, train_target_symbolic, val_target_symbolic = train_test_split(
                    symbols_convert, Y_data, test_size=0.2)


            elif self.task == "regression":
                # Fit the scaler to the training data and transform it
                X_data = self.scaler.fit_transform(data['X_data'])
                Y_data = data['Y_data']

                symbols = self.xabba.fit_transform(X_data, alphabet_set=alphabet_set)
                reconstruction = self.xabba.inverse_transform(symbols)
                # reconst_same_shape = self.xabba.recast_shape(reconstruction)  # recast into original shape

                symbols_convert = []
                for i_data in range(len(symbols)):
                    symbols_convert.append(listToString(list(symbols[i_data])))

                train_data_symbolic, val_data_symbolic, train_target_symbolic, val_target_symbolic = train_val_split(
                    symbols_convert, Y_data, test_size=0.2)

            elif self.task == "forecasting":
                # Fit the scaler to the training data and transform it
                X_data = self.scaler.fit_transform(data['X_data'])
                Y_data = data['Y_data']

                symbols = self.xabba.fit_transform(X_data, alphabet_set=alphabet_set)
                reconstruction = self.xabba.inverse_transform(symbols)
                # reconst_same_shape = self.xabba.recast_shape(reconstruction)  # recast into original shape

                symbols_convert = []
                for i_data in range(len(symbols)):
                    symbols_convert.append(listToString(list(symbols[i_data])))

                train_data_symbolic, val_data_symbolic, train_target_symbolic, val_target_symbolic = train_val_split(
                    symbols_convert, Y_data, test_size=0.2)




                # Fit the scaler to the training data and transform it
                X_data = self.scaler.fit_transform(data.X_data)
                Y_data = data.Y_data

                #############################################  Train Data  #############################################
                train_data = df_data[border1s[0]:border2s[0]]
                scaler.fit(train_data.values)
                train_data_transformed = scaler.transform(train_data)

                X_Train_data_patch = np.zeros(
                    [train_data_transformed.shape[0] - (seq_len_pre + seq_len_post), seq_len_pre,
                     train_data_transformed.shape[1]], dtype=float)
                Y_Train_data_patch = np.zeros(
                    [train_data_transformed.shape[0] - (seq_len_pre + seq_len_post), seq_len_post,
                     train_data_transformed.shape[1]], dtype=float)
                for i_data_patch in range(train_data_transformed.shape[0] - (seq_len_pre + seq_len_post)):
                    X_Train_data_patch[i_data_patch, :, :] = train_data_transformed[
                                                             i_data_patch:i_data_patch + seq_len_pre, :]
                    Y_Train_data_patch[i_data_patch, :, :] = train_data_transformed[
                                                             i_data_patch + seq_len_pre:i_data_patch + seq_len_pre + seq_len_post,
                                                             :]

                symbols_train_data = []
                symbols_train_data = qabba.fit_transform(X_Train_data_patch, alphabet_set=vocab_list, llm_split='Pre')
                reconstruction_train_data = qabba.inverse_transform(symbols_train_data)
                train_data_same_shape = qabba.recast_shape(reconstruction_train_data)  # recast into original shape

                symbols_train_target = []
                symbols_train_target, params_train_target = qabba.transform(Y_Train_data_patch, llm_split='Post')
                reconstruction_train_target = qabba.inverse_transform(symbols_train_target, params_train_target)
                train_target_same_shape = qabba.recast_shape(reconstruction_train_target,
                                                             recap_shape=Y_Train_data_patch.shape)  # recast into original shape

                print('##############################################################')
                print("The length of used symbols is:" + str(qabba.parameters.centers.shape[0]))

                train_data_symbolic = []
                for i_data in range(len(symbols_train_data)):
                    train_data_symbolic.append(listToString(list(symbols_train_data[i_data])))

                train_target_symbolic = []
                for i_data in range(len(symbols_train_target)):
                    train_target_symbolic.append(listToString(list(symbols_train_target[i_data])))

                arranged_seq = np.random.randint(len(train_data_symbolic), size=int(len(train_data_symbolic) * 0.2))

                val_data_symbolic = [train_data_symbolic[index] for index in arranged_seq]
                val_target_symbolic = [train_target_symbolic[index] for index in arranged_seq]

                data_TS = DatasetDict({
                    'train': Dataset.from_dict(
                        {'text_outputs': train_target_symbolic, 'text_inputs': train_data_symbolic}),
                    'val': Dataset.from_dict({'text_outputs': val_target_symbolic, 'text_inputs': val_data_symbolic}),
                })



            else:
                raise NotImplementedError("Method is not implemented, please contact the maintenance team.")

            data_TS = DatasetDict({
                'train': Dataset.from_dict({'text_outputs': train_target_symbolic, 'text_inputs': train_data_symbolic}),
                'val': Dataset.from_dict({'text_outputs': val_target_symbolic, 'text_inputs': val_data_symbolic})
            })

            #### Saving Scaler and ABBA
            output_scaler = open(str("../save/" + self.task  + "_Scaler" + "_save.pkl"), 'wb')

            str1 = pickle.dumps(self.scaler)
            output_scaler.write(str1)
            output_scaler.close()
            
            curr_loc = os.path.dirname(os.path.realpath(__file__))
            save_abba(self.xabba, str("../save/" + self.task  + "_ABBA" + "_save.pkl"))
            
            tokenized_train_dataset = data_TS['train'].map(self.build)
            tokenized_val_dataset = data_TS['val'].map(self.build)

            tokenized_train_dataset.set_format("torch")
            tokenized_val_dataset.set_format("torch")

        else:  ####  Inference Mode

            pass

        features = []
        targets = []


        if inference_mode:
            return features, None
        else:
            return tokenized_train_dataset, tokenized_val_dataset



    def model(self, pretrained_file = None, model_name = None, max_len = 2048) -> None:

        self.model_name = model_name
        self.max_len = max_len

        if pretrained_file is None:
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
            )
            accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

            """Model selection with parameters"""
            if self.quant_process is True:
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

                model_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_len=max_len,
                    padding_side="right",
                    truncation=True,
                    add_eos_token=True,
                )

                model_tokenizer.padding_side = 'right'
                model_tokenizer.pad_token = model_tokenizer.eos_token
                model_tokenizer.pad_token_id = model_tokenizer.eos_token_id

                model_input = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    quantization_config=quantization_config,
                    # device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

                # Data collator for padding a batch of examples to the maximum length seen in the batch
                model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)

                model_input.config.pad_token_id = model_input.config.eos_token_id
                model_input = prepare_model_for_kbit_training(model_input)

            else:

                model_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_len=max_len,
                    padding_side="right",
                    truncation=True,
                    add_eos_token=True,
                )

                model_tokenizer.padding_side = 'right'
                model_tokenizer.pad_token = model_tokenizer.eos_token
                model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
                model_input = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    # device_map="auto",
                    trust_remote_code=True,
                )
                # Data collator for padding a batch of examples to the maximum length seen in the batch
                self.model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)

                model_input.config.pad_token_id = model_input.config.eos_token_id
                model_input = prepare_model_for_kbit_training(model_input)

            #####   Loading LoRA config and PEFT model
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.lora_r,  # the dimension of the low-rank matrices
                lora_alpha=self.lora_alpha,  # scaling factor for LoRA activations vs pre-trained weight activations
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,  # dropout probability of the LoRA layers
                bias='none',  # wether to train bias weights, set to 'none' for attention layers
                modules_to_save=self.modules_to_save,
            )

            model_input = get_peft_model(model_input, lora_config)
            # Apply the accelerator. You can comment this out to remove the accelerator.
            model_input = accelerator.prepare_model(model_input)

            model_input.print_trainable_parameters()
            model_input = model_input.cuda()

            mistral_vocab = model_tokenizer.get_vocab()
            vocab_list = list(mistral_vocab.keys())

            print('##############################################################')
            print("The length of vocabulary list is:" + str(len(vocab_list)))

        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,  # Mistral, same as before
                quantization_config=bnb_config,  # Same quantization config as before
                # device_map="auto",
                trust_remote_code=True,
            )

            model_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=MAX_LENGTH,
                padding_side="right",
                truncation=True,
                add_eos_token=True,
            )

            model_tokenizer.padding_side = 'right'
            model_tokenizer.pad_token = model_tokenizer.eos_token
            print(len(model_tokenizer))

            mistral_vocab = model_tokenizer.get_vocab()
            model_input = PeftModel.from_pretrained(model, pretrained_file)

            # ft_model = PeftModel.from_pretrained(model, 'llama2-7B-ts-finetune-ETTh1-r16-Pre168-Post168' + "/" + peft_file[0])



        # vocab_list_new = []
        # for i_vac in vocab_list:
        #     if '▁' in i_vac:
        #         vocab_list_new.append(i_vac.replace('▁', ''))
        #     else:
        #         vocab_list_new.append(i_vac)

        self.model_tokenizer = model_tokenizer
        return model_input, model_tokenizer



    def train(self, model_input, num_epochs, output_dir, train_dataset, val_dataset):
        """Train with validation"""
        if torch.cuda.device_count() > 1:  # If more than 1 GPU
            model.is_parallelizable = True
            model.model_parallel = True


        run_name = self.task + "-" + self.model_name

        trainer = Trainer(
            model=model_input,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=TrainingArguments(
                output_dir=output_dir,
                warmup_steps=5,
                per_device_train_batch_size=4,
                gradient_checkpointing=True,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,  # Want about 10x smaller than the Mistral learning rate
                bf16=False,
                optim="paged_adamw_8bit",
                num_train_epochs=num_epochs,
                weight_decay=0.00005,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                logging_dir="./logs",
                load_best_model_at_end=True,
                report_to="wandb",
                save_total_limit=1,
                do_eval=True,  # Perform evaluation at the end of training
                # report_to="wandb",           # Comment this out if you don't want to use weights & baises
                run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"  # Name of the W&B run (optional)
            ),
            data_collator=DataCollatorForLanguageModeling(self.model_tokenizer, mlm=False),
        )

        model_input.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()


    def inference(self, data):
        """Inference """
        (processed_data, _) = self.process(data, inference_mode=True)

        return self.model(processed_data)
    


    def save_model(self, parameter_types="ABBA"):
        pass


    def load_model(self):
        pass