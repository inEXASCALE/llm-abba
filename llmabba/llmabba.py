from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from .abba import XABBA
import pickle
import os
import torch
from datasets.dataset_dict import DatasetDict
from peft import PeftModel
from .utils.fundamentals import *

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
                       target_modules = [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                            "lm_head",
                        ],

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
        self.target_modules = target_modules

        self.modules_to_save = modules_to_save
        self.lora_dropout = lora_dropout




    def process(self, data, task, seg_length=16, 
                seq_len_pre = 168,
                seq_len_post = 168, data_name="ETTh1"):

        if task == "classification":
            if data_name == 'eeg-eye-state':
                scaler = StandardScaler()
                clean_data_frame_second_normalized = scaler.fit_transform(clean_data_frame_second)
                data_labels = data['eyeDetection'].to_numpy()

                total_seg_num = int(len(data_labels) / seg_length)

                X_data = np.zeros([total_seg_num, seg_length, 14], dtype=float)
                y_data = np.zeros([total_seg_num], dtype=int)

                seg_num = 0
                for i_seg in range(total_seg_num):
                    X_data[seg_num, :, :] = clean_data_frame_second_normalized[i_seg * seg_length:(i_seg + 1) * seg_length, :-1]
                    temp_a = data_labels[i_seg * seg_length:(i_seg + 1) * seg_length]
                    # print(temp_a)
                    if np.mean(temp_a) > 0.5:
                        y_data[seg_num] = 1
                    seg_num += 1

                from sklearn.utils.class_weight import compute_class_weight
                label_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_data), y=y_data)

                
                jabba = XABBA(tol=self.abba_tol, init=self.abba_init, alpha=self.abba_alpha, scl=self.abba_scl verbose=0)

                symbols = jabba.fit_transform(X_data, alphabet_set=vocab_list)
                reconstruction = jabba.inverse_transform(symbols)
                reconst_same_shape = jabba.recast_shape(reconstruction)  # recast into original shape

                print("The length of one symbols element is:" + str(len(symbols[10])))
                print("The length of used symbols is:" + str(jabba.parameters.centers.shape[0]))

                symbols_convert = []
                for i_data in range(len(symbols)):
                    # print(i_data)
                    symbols_convert.append(listToString(list(symbols[i_data])))

                train_data_symbolic, test_data_symbolic, train_target_symbolic, test_target_symbolic = train_test_split(
                    symbols_convert, y_data, test_size=0.2)

            elif data_name == 'ptbdb':


                jabba = XABBA(tol=self.abba_tol, init=self.abba_init, alpha=self.abba_alpha, scl=self.abba_scl verbose=0)

                # Initialize the StandardScaler
                scaler = StandardScaler()

                # Fit the scaler to the training data and transform it
                X_data = scaler.fit_transform(X_data)
                symbols = jabba.fit_transform(X_data, alphabet_set=vocab_list)
                print("The length of one symbols element is:" + str(len(symbols[100])))
                print("The length of used symbols is:" + str(jabba.parameters.centers.shape[0]))

                symbols_convert = []
                for i_data in range(len(symbols)):
                    symbols_convert.append(listToString(list(symbols[i_data])))

                train_data_symbolic, test_data_symbolic, train_target_symbolic, test_target_symbolic = train_test_split(
                    symbols_convert, y_data, test_size=0.2)

            elif data_name == 'fnirs':

                ###############   fnirs data   ###############

                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                X_data_temp = scaler.fit_transform(X_data_temp)
                X_data = np.reshape(X_data_temp, (X_data.shape[0], X_data.shape[1], X_data.shape[2]))

                # X_data = np.transpose(X_data, (0, 2, 1))
                jabba = XABBA(tol=self.abba_tol, init=self.abba_init, alpha=self.abba_alpha, scl=self.abba_scl verbose=0)

                symbols = jabba.fit_transform(X_data, alphabet_set=vocab_list)
                reconstruction = jabba.inverse_transform(symbols)
                print("The length of one symbols element is:" + str(len(symbols[100])))
                print("The length of used symbols is:" + str(jabba.parameters.centers.shape[0]))

                symbols_convert = []
                for i_data in range(len(symbols)):
                    symbols_convert.append(listToString(list(symbols[i_data])))

                from sklearn.utils.class_weight import compute_class_weight
                label_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_data), y=y_data)

                train_data_symbolic_temp, test_data_symbolic, train_target_symbolic, test_target_symbolic = \
                    symbols_convert[:train_length], symbols_convert[train_length:], y_data[:train_length], y_data[train_length:]

                train_data_symbolic = []
                arranged_seq = np.random.randint(len(train_target_symbolic), size=len(train_target_symbolic))
                for i_arranged_seq in arranged_seq:
                    train_data_symbolic.append(train_data_symbolic_temp[i_arranged_seq])
                train_target_symbolic = train_target_symbolic[arranged_seq]

            elif data_name == 'mitbih':

                # Initialize the StandardScaler
                scaler = StandardScaler()

                # Fit the scaler to the training data and transform it
                X_data = scaler.fit_transform(X_data)

                jabba = XABBA(tol=self.abba_tol, init=self.abba_init, alpha=self.abba_alpha, scl=self.abba_scl verbose=1)

                symbols = jabba.fit_transform(X_data, alphabet_set=vocab_list_new)
                # reconstruction = jabba.inverse_transform(symbols)
                print("The length of one symbols element is:" + str(len(symbols[100])))
                print("The length of used symbols is:" + str(jabba.parameters.centers.shape[0]))


                symbols_convert = []
                for i_data in range(len(symbols)):
                    # print(i_data)
                    symbols_convert.append(listToString(list(symbols[i_data])))

                from sklearn.utils.class_weight import compute_class_weight
                label_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_data), y=y_data)

                train_data_symbolic, test_data_symbolic, train_target_symbolic, test_target_symbolic = train_test_split(
                    symbols_convert, y_data, test_size=0.2)

            data_TS = DatasetDict({
                'train': Dataset.from_dict({'text_outputs': train_target_symbolic, 'text_inputs': train_data_symbolic}),
                'val': Dataset.from_dict({'text_outputs': test_target_symbolic, 'text_inputs': test_data_symbolic})
            })

            
            output_scaler = open(str(model_name + "_" + data_name + "-Scaler_Pre" + "_save.pkl"), 'wb')
            output_jabba = open(str(model_name + "_" + data_name + "-ABBA_Pre" + "_save.pkl"), 'wb')

            str1 = pickle.dumps(scaler)
            output_scaler.write(str1)
            output_scaler.close()

            str2 = pickle.dumps(qabba)
            output_jabba.write(str2)
            output_jabba.close()

        elif task == "regression":
            
            ###############   Monash Regression data   ###############
            data_scaled = process_data(data_all, normalise=norm, min_len=min_len)

            jabba = XABBA(tol=self.abba_tol, init=self.abba_init, alpha=self.abba_alpha, scl=self.abba_scl verbose=0)
            
            # symbols = jabba.fit_transform(data_scaled)
            symbols = jabba.fit_transform(data_scaled, alphabet_set=vocab_list)
            reconstruction = jabba.inverse_transform(symbols)
            reconst_same_shape = jabba.recast_shape(reconstruction)  # recast into original shape
 
            print("[{}] Task: {}".format(model_name, data_name))
            print("The length of used symbols is:" + str(jabba.parameters.centers.shape[0]))


            symbols_convert = []
            for i_data in range(len(symbols)):
                symbols_convert.append(listToString(list(symbols[i_data])))

            train_data_symbolic, test_data_symbolic, train_target_symbolic, test_target_symbolic = \
                symbols_convert[:train_test_split[0]], symbols_convert[train_test_split[0]:], \
                target_scaled[:train_test_split[0]], target_scaled[train_test_split[0]:]

            data_TS = DatasetDict({
                'train': Dataset.from_dict({'text_outputs': train_target_symbolic, 'text_inputs': train_data_symbolic}),
                'val': Dataset.from_dict({'text_outputs': test_target_symbolic, 'text_inputs': test_data_symbolic})
            })

            
        elif task == "forecasting":

            qabba = XABBA(tol=0.01, init='agg', alpha=0.01, scl=3, verbose=0) 

            scaler = StandardScaler()
            train_data = data[border1s[0]:border2s[0]]
            scaler.fit(train_data.values)
            train_data_transformed = scaler.transform(train_data)

            X_Train_data_patch = np.zeros([train_data_transformed.shape[0] - (seq_len_pre + seq_len_post), seq_len_pre, train_data_transformed.shape[1]], dtype=float)
            Y_Train_data_patch = np.zeros([train_data_transformed.shape[0] - (seq_len_pre + seq_len_post), seq_len_post, train_data_transformed.shape[1]], dtype=float)
            for i_data_patch in range(train_data_transformed.shape[0] - (seq_len_pre + seq_len_post)):
                X_Train_data_patch[i_data_patch, :, :] = train_data_transformed[i_data_patch:i_data_patch + seq_len_pre, :]
                Y_Train_data_patch[i_data_patch, :, :] = train_data_transformed[i_data_patch + seq_len_pre:i_data_patch + seq_len_pre + seq_len_post, :]

            symbols_train_data = []
            symbols_train_data = qabba.fit_transform(X_Train_data_patch, alphabet_set=vocab_list, llm_split='Pre')
            reconstruction_train_data = qabba.inverse_transform(symbols_train_data)
            train_data_same_shape = qabba.recast_shape(reconstruction_train_data)  # recast into original shape

            symbols_train_target = []
            symbols_train_target, params_train_target = qabba.transform(Y_Train_data_patch, llm_split='Post')
            reconstruction_train_target = qabba.inverse_transform(symbols_train_target, params_train_target)
            train_target_same_shape = qabba.recast_shape(reconstruction_train_target, recap_shape=Y_Train_data_patch.shape)  # recast into original shape

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
                'train': Dataset.from_dict({'text_outputs': train_target_symbolic, 'text_inputs': train_data_symbolic}),
                'val': Dataset.from_dict({'text_outputs': val_target_symbolic, 'text_inputs': val_data_symbolic}),
            })

            from sklearn.metrics import mean_squared_error, mean_absolute_error

            train_length = 1

            Y_recons_all = np.zeros(([train_length, 7 * seq_len_post]), dtype=float)
            Y_true_all = np.zeros(([train_length, 7 * seq_len_post]), dtype=float)

            for i_reconst in range(train_length):


                Y_true_pre = scaler.inverse_transform(X_Train_data_patch[i_reconst, :, :])
                Y_true_post = scaler.inverse_transform(Y_Train_data_patch[i_reconst, :, :])

                Y_recons_pre = scaler.inverse_transform(train_data_same_shape[i_reconst, :, :])
                Y_recons_post = scaler.inverse_transform(train_target_same_shape[i_reconst, :, :])

                Y_recons = np.reshape(Y_recons_post, (1, 7 * seq_len_post))
                Y_true = np.reshape(Y_true_post, (1, 7 * seq_len_post))


                Y_true_all[i_reconst, :] = Y_true
                Y_recons_all[i_reconst, :] = Y_recons

            print(mean_squared_error(Y_true_all, Y_recons_all))
            print(mean_absolute_error(Y_true_all, Y_recons_all))
            
            
        else:
            print("No data here!")


        tokenized_train_dataset = data_TS['train'].map(generate_and_tokenize_prompt)
        tokenized_val_dataset = data_TS['val'].map(generate_and_tokenize_prompt)

        tokenized_train_dataset.set_format("torch")
        tokenized_val_dataset.set_format("torch")

        model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)

        if torch.cuda.device_count() > 1: # If more than 1 GPU
            model.is_parallelizable = True
            model.model_parallel = True


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



        # ft_model = PeftModel.from_pretrained(model, 'llama2-7B-ts-finetune-ETTh1-r16-Pre168-Post168' + "/" + peft_file[0])

            