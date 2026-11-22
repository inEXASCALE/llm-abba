from datetime import datetime
import math
from os.path import commonprefix
from pathlib import Path
import pickle

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
from packaging.version import Version
import pandas as pd
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding, Trainer, TrainingArguments,
    __version__ as transformers_version,
)

from .utils.fundamentals import mean
from .utils.regressor_tools import process_data
from .xabba import XABBA

def save_abba(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_abba(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def ListToString(symbols):
    return " ".join(symbols)


def StringToList(text):
    return text.split()


def _four_bit_load_options():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    else:
        dtype = torch.float32

    dtype_argument = "dtype" if Version(transformers_version).release >= (4, 56) else "torch_dtype"
    return {
        dtype_argument: dtype,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        ),
    }


def _balanced_classification_dataset(dataset, seed):
    if "text_outputs" not in dataset.column_names:
        raise ValueError("Balanced sampling requires a classification dataset returned by process().")

    groups = {}
    for index, label in enumerate(dataset.with_format("python")["text_outputs"]):
        groups.setdefault(str(label), []).append(index)
    if len(groups) <= 1:
        return dataset

    rng = np.random.default_rng(seed)
    per_class, remainder = divmod(len(dataset), len(groups))
    group_indices = list(groups.values())
    selected = []
    for offset, group in enumerate(rng.permutation(len(group_indices))):
        indices = group_indices[group]
        count = per_class + (offset < remainder)
        if count <= len(indices):
            selected.extend(rng.choice(indices, size=count, replace=False))
        else:
            selected.extend(indices)
            extra = count - len(indices)
            selected.extend(rng.choice(indices, size=extra, replace=extra > len(indices)))
    return dataset.select(rng.permutation(selected).tolist())


def cross_correlation(x, y):
    # Calculate means
    x_mean = mean(x)
    y_mean = mean(y)

    # Calculate numerator
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))

    # Calculate denominators
    x_sq_diff = sum((a - x_mean) ** 2 for a in x)
    y_sq_diff = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_sq_diff * y_sq_diff)
    correlation = numerator / denominator

    return correlation


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
        self.max_len = 2048


    def tokenize(self, prompt_input):
        result = self.model_tokenizer(
            prompt_input,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        result["labels"] = [
            token if mask else -100
            for token, mask in zip(result["input_ids"], result["attention_mask"])
        ]
        return result

    def build(self, data_point):
        class_tokens = getattr(getattr(self, "xabba", None), "classification_label_tokens", None)
        if class_tokens is not None:
            tokenizer = self.model_tokenizer
            prefix = self._classification_prompt_ids(self.prompt, data_point["text_inputs"], tokenizer)
            answer = [*class_tokens[str(data_point["text_outputs"])], tokenizer.eos_token_id]
            padding = self.xabba.classification_max_len - len(prefix) - len(answer)
            return {
                "input_ids": prefix + answer + [tokenizer.pad_token_id] * padding,
                "attention_mask": [1] * (len(prefix) + len(answer)) + [0] * padding,
                "labels": [-100] * len(prefix) + answer + [-100] * padding,
            }
        prompt = self._format_prompt(self.prompt, data_point["text_inputs"])
        return self.tokenize(f'{prompt} {data_point["text_outputs"]}')

    def _classification_prompt_ids(self, prompt, symbols, tokenizer):
        longest_label = max(map(len, self.xabba.classification_label_tokens.values()))
        budget = self.xabba.classification_max_len - longest_label - 1

        def encode(series):
            if getattr(tokenizer, "chat_template", None):
                message = {"role": "user", "content": f"{prompt}\n### Symbolic Series: {series}"}
                return tokenizer.apply_chat_template(
                    [message], tokenize=True, add_generation_prompt=True, return_dict=False,
                    tokenizer_kwargs={"verbose": False},
                )
            text = self._format_prompt(prompt, series)
            return tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]

        ids = encode(symbols)
        if ids and len(ids) <= budget:
            return ids

        # Truncate only the series so the instruction and assistant boundary stay intact.
        best = encode("")
        if not best or len(best) > budget:
            raise ValueError("max_len is too short for the classification prompt and its answer.")
        pieces = symbols.split()
        low, high = 1, len(pieces)
        while low <= high:
            middle = (low + high) // 2
            candidate = encode(" ".join(pieces[:middle]))
            if len(candidate) <= budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _predict_classification_batch(self, prefixes, ft_model, tokenizer):
        """Decode known labels for a batch of equally sized prompt prefixes."""
        candidates = self.xabba.classification_label_tokens
        if len(candidates) == 1:
            return [next(iter(candidates))] * len(prefixes)

        shared = commonprefix(list(candidates.values()))
        labels = {tuple(tokens[len(shared):]): label for label, tokens in candidates.items()}
        eos = tokenizer.eos_token_id
        allowed = {}
        for tokens in labels:
            for index, token in enumerate(tokens):
                allowed.setdefault(tokens[:index], set()).add(token)
            allowed.setdefault(tokens, set()).add(eos)
        allowed = {prefix: sorted(tokens) for prefix, tokens in allowed.items()}

        inputs = torch.tensor([prefix + list(shared) for prefix in prefixes], dtype=torch.long,
                              device=getattr(ft_model, "device", None))
        prompt_length = inputs.shape[-1]

        def allowed_tokens(batch_index, sequence):
            continuation = tuple(sequence[prompt_length:].tolist())
            return [eos] if eos in continuation else allowed[continuation]

        generated = ft_model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            do_sample=False, num_beams=1, max_new_tokens=max(map(len, labels)),
            eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
        sequences = getattr(generated, "sequences", generated)
        results = []
        for continuation in sequences[:, prompt_length:].tolist():
            if eos in continuation:
                continuation = continuation[:continuation.index(eos)]
            label = labels.get(tuple(continuation))
            if label is None:
                raise RuntimeError("The model generated tokens outside the trained classification labels.")
            results.append(label)
        return results

    @staticmethod
    def _format_prompt(prompt, symbols):
        return f"{prompt}\n### Symbolic Series: {symbols}\n### Results:"


    def process(self, project_name, data, task, prompt, model_tokenizer,
                seq_len_pre=24, scalar="z-score", seq_len_post=24, alphabet_set=None, draw_ts=False):
        """Fit preprocessing on training data and return tokenized train and validation datasets."""
        if scalar == "min-max":
            self.scaler = preprocessing.MinMaxScaler()
        elif scalar == "z-score":
            self.scaler = preprocessing.StandardScaler()
        else:
            raise ValueError(f"Unknown scaler: {scalar!r}")

        if alphabet_set is None:
            special_tokens = set(getattr(model_tokenizer, "all_special_tokens", ()))
            alphabet_set = [
                token for token in model_tokenizer.get_vocab()
                if token.isprintable() and not any(char.isspace() for char in token)
                and token not in special_tokens and not any(marker in token for marker in "▁ĠĊ")
            ]
            if not alphabet_set:
                alphabet_set = 0
        elif isinstance(alphabet_set, list):
            if any(not isinstance(token, str) or not token or any(char.isspace() for char in token)
                   for token in alphabet_set):
                raise ValueError("Custom alphabet symbols must be nonempty strings without whitespace.")

        self.project_name = project_name
        self.task = task
        self.alphabet_set = alphabet_set
        self.prompt = prompt
        self.model_tokenizer = model_tokenizer
        self.seq_len_pre = seq_len_pre
        self.seq_len_post = seq_len_post
        self.xabba = XABBA(
            tol=self.abba_tol, init=self.abba_init, alpha=self.abba_alpha, scl=self.abba_scl,
            bits_for_len=self.bits_for_len, bits_for_inc=self.bits_for_inc, verbose=self.abba_verbose,
        )

        if task == "forecasting":
            train_values, val_values, train_targets, val_targets = self._forecast_windows(data)
        elif task in {"classification", "regression"}:
            train_values, val_values, train_targets, val_targets = train_test_split(
                data["X_data"], data["Y_data"], test_size=0.2,
            )
            train_values = self._supervised_values(train_values)
            series_length = train_values.shape[1] if train_values.ndim == 3 else None
            val_values = self._supervised_values(val_values, series_length)
            train_values = self._scale_batch(train_values, fit=True)
            val_values = self._scale_batch(val_values)
        else:
            raise ValueError(f"Unknown task: {task!r}")

        if task == "classification":
            self.xabba.fit(train_values, alphabet_set=alphabet_set)
            train_symbols, _ = self.xabba.transform(train_values)
        else:
            train_symbols = self.xabba.fit_transform(train_values, alphabet_set=alphabet_set)
        val_symbols, _ = self.xabba.transform(val_values)

        if task == "classification":
            if model_tokenizer.eos_token_id is None or model_tokenizer.pad_token_id is None:
                raise ValueError("Classification requires a tokenizer with EOS and padding tokens.")
            label_tokens = {
                label: tuple(model_tokenizer(f" {label}", add_special_tokens=False)["input_ids"])
                for label in dict.fromkeys(map(str, data["Y_data"]))
            }
            if (any(not tokens or model_tokenizer.eos_token_id in tokens for tokens in label_tokens.values())
                    or len(set(label_tokens.values())) != len(label_tokens)):
                raise ValueError("Classification labels must encode to distinct, nonempty token sequences without EOS.")
            self.xabba.classification_label_tokens = label_tokens
            self.xabba.classification_max_len = self.max_len

        if task == "forecasting":
            train_targets, _ = self.xabba.transform(train_targets, llm_split="Post")
            val_targets, _ = self.xabba.transform(val_targets, llm_split="Post")
            train_targets = [ListToString(symbols) for symbols in train_targets]
            val_targets = [ListToString(symbols) for symbols in val_targets]
            self.xabba.forecast_uses_history_anchor = True
            self.xabba.forecast_horizon = seq_len_post

        if draw_ts:
            reconstruction = self.xabba.inverse_transform(train_symbols[:1], self.xabba.start_set[:1])
            reconstruction = self.xabba.recast_shape(reconstruction, recap_shape=train_values.shape)
            self._plot_series(train_values[0], reconstruction[0])

        train_dataset = self._build_dataset(train_symbols, train_targets)
        val_dataset = self._build_dataset(val_symbols, val_targets)

        directory = Path("../save") / project_name
        directory.mkdir(parents=True, exist_ok=True)
        save_abba(self.scaler, directory / f"{task}_Scaler_save.pkl")
        save_abba(self.xabba, directory / f"{task}_ABBA_save.pkl")
        return train_dataset, val_dataset

    @staticmethod
    def _supervised_values(values, series_length=None):
        if isinstance(values, pd.DataFrame) and pd.api.types.is_list_like(values.iat[0, 0]):
            if series_length is None:
                series_length = min(len(series) for row in values.itertuples(index=False, name=None) for series in row)
            values, _ = process_data(values, min_len=series_length)
        return np.asarray(values, dtype=float)

    def _scale_batch(self, values, fit=False):
        if values.ndim not in (2, 3):
            raise ValueError("Expected a two- or three-dimensional batch of time series.")
        scale = self.scaler.fit_transform if fit else self.scaler.transform
        return scale(values.reshape(-1, values.shape[-1])).reshape(values.shape)

    def _forecast_windows(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError("Forecasting data must have shape (time, features).")
        if self.seq_len_pre < 1 or self.seq_len_post < 1:
            raise ValueError("Forecasting input and output lengths must be positive.")
        window_size = self.seq_len_pre + self.seq_len_post
        if len(values) < 2 * window_size:
            raise ValueError(f"At least {2 * window_size} time points are required for separate train and validation windows.")

        split = min(max(int(len(values) * 0.8), window_size), len(values) - window_size)
        self.scaler.fit(values[:split])

        def make_windows(block):
            scaled = self.scaler.transform(block)
            windows = np.lib.stride_tricks.sliding_window_view(scaled, window_size, axis=0).swapaxes(1, 2)
            inputs = windows[:, :self.seq_len_pre]
            targets = windows[:, self.seq_len_pre:].reshape(len(windows), -1)
            # The last observed value anchors reconstruction of the first future value.
            targets = np.column_stack((inputs[:, -1, -1], targets))
            return inputs, targets

        train_values, train_targets = make_windows(values[:split])
        val_values, val_targets = make_windows(values[split:])
        return train_values, val_values, train_targets, val_targets

    def _build_dataset(self, symbols, targets):
        dataset = Dataset.from_dict({
            "text_inputs": [ListToString(sequence) for sequence in symbols],
            "text_outputs": list(targets),
        }).map(self.build)
        dataset.set_format("torch")
        return dataset

    @staticmethod
    def _plot_series(expected, reconstructed):
        plt.plot(expected.reshape(len(expected), -1)[:, 0], label="Ground Truth")
        plt.plot(reconstructed.reshape(len(reconstructed), -1)[:, 0], label="Reconstruction")
        plt.legend()
        plt.show()

    def model(self, peft_file = None, model_name = None, max_len = 2048) -> None:

        self.model_name = model_name
        self.max_len = max_len

        if peft_file is None:
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
            )
            accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

        model_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=max_len,
            padding_side="right",
            truncation=True,
            add_eos_token=True,
        )
        model_tokenizer.padding_side = "right"
        model_tokenizer.pad_token = model_tokenizer.eos_token
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id

        load_options = _four_bit_load_options() if self.quant_process else {}
        model_input = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **load_options)

        if peft_file is None:
            if not self.quant_process:
                self.model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)

            model_input.config.pad_token_id = model_input.config.eos_token_id
            model_input = prepare_model_for_kbit_training(model_input)

            #####   Loading LoRA config and PEFT model
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
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

        else:
            model_input = PeftModel.from_pretrained(model_input, peft_file)

        self.model_tokenizer = model_tokenizer
        return model_input, model_tokenizer



    def train(self, model_input, num_epochs, output_dir, train_dataset, val_dataset, *,
              balance_classes=False, sampling_seed=42):
        """Train with validation and optional class-balanced sampling of the training set."""
        if balance_classes:
            if self.task != "classification":
                raise ValueError("Balanced sampling is available for classification only.")
            train_dataset = _balanced_classification_dataset(train_dataset, sampling_seed)

        if torch.cuda.device_count() > 1:  # If more than 1 GPU
            model_input.is_parallelizable = True
            model_input.model_parallel = True

        run_name = self.task + "-" + self.model_name
        checkpointing_argument = (
            "activation_checkpointing" if hasattr(TrainingArguments, "activation_checkpointing")
            else "gradient_checkpointing"
        )

        trainer = Trainer(
            model=model_input,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=TrainingArguments(
                output_dir=output_dir,
                warmup_steps=5,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                bf16=False,
                optim="paged_adamw_8bit" if self.quant_process else "adamw_torch",
                num_train_epochs=num_epochs,
                weight_decay=0.00005,
                eval_strategy="epoch",  # Updated from previous fix
                save_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=True,
                report_to="wandb",
                save_total_limit=1,
                do_eval=True,
                run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
                **{checkpointing_argument: True},
            ),
            data_collator=DataCollatorWithPadding(tokenizer=self.model_tokenizer),  # Changed
        )

        model_input.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()


    def inference(self, project_name, data, task, prompt, ft_model, model_tokenizer, scalar="z-score",
                  seq_len_pre=24, seq_len_post=None, llm_max_length=1024, llm_repetition_penalty=1.7,
                  llm_temperature=0.0, llm_max_new_tokens=256, draw_ts=False):
        """Generate a result for each sample or reconstruct a forecast window."""
        if task not in {"classification", "regression", "forecasting"}:
            raise ValueError(f"Unknown task: {task!r}")
        if prompt is None:
            raise ValueError("A prompt is required for inference.")

        directory = Path("../save") / project_name
        self.xabba = load_abba(directory / f"{task}_ABBA_save.pkl")
        self.scaler = load_abba(directory / f"{task}_Scaler_save.pkl")

        if task == "forecasting":
            values = np.asarray(data, dtype=float)
            if values.ndim != 2:
                raise ValueError("Forecasting data must have shape (time, features).")
            fitted_shape = self.xabba.recap_shape
            history_length = fitted_shape[1] if fitted_shape is not None else seq_len_pre
            if len(values) != history_length:
                raise ValueError(f"Expected {history_length} time points in the forecast history.")
            scaled_history = self.scaler.transform(values)
            data_scaled = scaled_history[np.newaxis, ...]
        else:
            fitted_shape = self.xabba.recap_shape
            values = self._supervised_values(data, fitted_shape[1] if fitted_shape is not None else None)
            if values.ndim == 1 or (values.ndim == 2 and fitted_shape is not None):
                values = values[np.newaxis, ...]
            data_scaled = self._scale_batch(values)

        sequences, _ = self.xabba.transform(data_scaled)
        if task == "classification" and hasattr(self.xabba, "classification_label_tokens"):
            outputs = []
            for symbols in sequences:
                prefix = self._classification_prompt_ids(prompt, ListToString(symbols), model_tokenizer)
                outputs.extend(self._predict_classification_batch([prefix], ft_model, model_tokenizer))
            return outputs[0] if len(outputs) == 1 else outputs

        options = {"repetition_penalty": llm_repetition_penalty, "do_sample": llm_temperature > 0}
        if llm_temperature > 0:
            options["temperature"] = llm_temperature
        if llm_max_new_tokens is None:
            options["max_length"] = llm_max_length
        else:
            options["max_new_tokens"] = llm_max_new_tokens
        device = getattr(ft_model, "device", None)

        outputs = []
        for symbols in sequences:
            test_prompt = self._format_prompt(prompt, ListToString(symbols))
            inputs = model_tokenizer(test_prompt, return_tensors="pt")
            if device is not None:
                inputs = inputs.to(device)
            prompt_length = inputs["input_ids"].shape[-1]
            generated = ft_model.generate(**inputs, **options)
            sequences_out = getattr(generated, "sequences", generated)
            outputs.append(model_tokenizer.decode(sequences_out[0, prompt_length:], skip_special_tokens=True).strip())

        if task != "forecasting":
            return outputs[0] if len(outputs) == 1 else outputs

        if seq_len_post is None:
            seq_len_post = getattr(self.xabba, "forecast_horizon", 24)
        if seq_len_post < 1:
            raise ValueError("The forecast length must be positive.")

        tokens = StringToList(outputs[0])
        alphabet = set(self.xabba.parameters.alphabets)
        previous = next((token for token in tokens if token in alphabet), None)
        if previous is None:
            raise ValueError("The generated forecast contains no recognized ABBA symbols.")
        forecast_symbols = []
        for token in tokens:
            if token in alphabet:
                previous = token
            forecast_symbols.append(previous)

        reconstruction = self.xabba.inverse_transform([forecast_symbols], [scaled_history[-1, -1]])[0]
        if getattr(self.xabba, "forecast_uses_history_anchor", False):
            reconstruction = reconstruction[1:]
        n_features = scaled_history.shape[1]
        reconstructed = self.xabba.recast_shape([reconstruction], recap_shape=(1, seq_len_post, n_features))[0]
        forecast = self.scaler.inverse_transform(reconstructed)
        if draw_ts:
            plt.plot(np.concatenate((values[:, 0], forecast[:, 0])), label="Forecast")
            plt.axvline(len(values) - 1)
            plt.legend()
            plt.show()
        return forecast
