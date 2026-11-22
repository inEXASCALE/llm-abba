import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import BatchEncoding
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"LLM dependencies unavailable: {error.name}") from error

from llmabba import LLMABBA
from llmabba.llm_abba import ListToString, StringToList, _four_bit_load_options
from llmabba.llmabba import LLMABBA as LegacyLLMABBA
from llmabba.utils.regressor_tools import calculate_regression_metrics
from llmabba.xabba import ShapeWarning


class ToyTokenizer:
    all_special_tokens = ("<pad>",)
    eos_token_id = 3
    pad_token_id = 3

    def __init__(self):
        self.prompts = []
        self.continuation = ""

    def get_vocab(self):
        return {"rise": 0, "fall": 1, "flat": 2, "<pad>": 3, "▁▁": 4}

    def __call__(self, prompt, max_length=None, return_tensors=None, **kwargs):
        if return_tensors == "pt":
            self.prompts.append(prompt)
            return BatchEncoding({"input_ids": torch.tensor([[8, 9]]), "attention_mask": torch.tensor([[1, 1]])})
        if kwargs.get("add_special_tokens") is False:
            if prompt.strip().isdigit():
                return {"input_ids": [20 + int(prompt)]}
            self.prompts.append(prompt)
            return {"input_ids": [8, 9]}
        return {
            "input_ids": [8, 9, 3] + [3] * (max_length - 3),
            "attention_mask": [1, 1, 1] + [0] * (max_length - 3),
        }

    def decode(self, ids, **kwargs):
        if list(ids) != [10]:
            raise AssertionError("The tokenizer should receive only the generated tokens.")
        return self.continuation


class ToyGenerator:
    device = torch.device("cpu")

    def __init__(self, tokenizer, continuations):
        self.tokenizer = tokenizer
        self.continuations = iter(continuations)

    def generate(self, input_ids, **kwargs):
        self.tokenizer.continuation = next(self.continuations)
        return torch.cat((input_ids, torch.tensor([[10]])), dim=1)


class ToyChatTokenizer(ToyTokenizer):
    chat_template = "toy"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict, tokenizer_kwargs):
        assert tokenize and add_generation_prompt and not return_dict
        assert [message["role"] for message in messages] == ["user"]
        series = messages[0]["content"].rsplit("### Symbolic Series:", 1)[1]
        return [1, 8, *([12] * len(series.split())), 9]


class ToyClassifier:
    device = torch.device("cpu")

    def __init__(self, labels):
        self.labels = iter(labels)

    def generate(self, input_ids, prefix_allowed_tokens_fn, **kwargs):
        tokens = []
        for index, prefix in enumerate(input_ids):
            output = 20 + next(self.labels)
            allowed = prefix_allowed_tokens_fn(index, prefix)
            assert 3 not in allowed and output in allowed
            tokens.append([output])
        return torch.cat((input_ids, torch.tensor(tokens)), dim=1)


class LLMABBARegressionTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        current = Path(temp.name) / "work"
        current.mkdir()
        os.chdir(current)
        self.tokenizer = ToyTokenizer()
        self.model = LLMABBA()
        self.model.max_len = 12

    def test_legacy_import_and_padding_mask(self):
        self.assertIs(LLMABBA, LegacyLLMABBA)
        self.model.model_tokenizer = self.tokenizer
        result = self.model.tokenize("classify")
        self.assertEqual(result["labels"], [8, 9, 3] + [-100] * 9)

    def test_classification_preserves_chat_prefix_and_supervises_only_answer(self):
        tokenizer = ToyChatTokenizer()
        self.model.model_tokenizer = tokenizer
        self.model.prompt = "Classify"
        self.model.xabba = SimpleNamespace(
            classification_label_tokens={"0": (20,), "1": (21,)}, classification_max_len=8,
        )
        symbols = "A B C D E F G H"
        prefix = self.model._classification_prompt_ids("Classify", symbols, tokenizer)
        self.assertEqual(prefix, [1, 8, 12, 12, 12, 9])
        result = self.model.build({"text_inputs": symbols, "text_outputs": 1})
        self.assertEqual(result["input_ids"], prefix + [21, 3])
        self.assertEqual(result["labels"], [-100] * len(prefix) + [21, 3])

        padded = self.model.build({"text_inputs": "A", "text_outputs": 0})
        self.assertEqual(padded["labels"], [-100] * 4 + [20, 3, -100, -100])
        self.assertEqual(padded["attention_mask"], [1] * 6 + [0, 0])

    def test_classification_batch_allows_padding_after_a_shorter_label(self):
        self.model.xabba = SimpleNamespace(classification_label_tokens={
            "short": (20,), "long": (20, 21, 22),
        })

        def generate(input_ids, prefix_allowed_tokens_fn, **kwargs):
            self.assertEqual(input_ids.tolist(), [[8, 9, 20], [8, 9, 20]])
            self.assertEqual(prefix_allowed_tokens_fn(0, input_ids[0]), [3, 21])
            first_step = torch.cat((input_ids, torch.tensor([[3], [21]])), dim=1)
            self.assertEqual(prefix_allowed_tokens_fn(0, first_step[0]), [3])
            self.assertEqual(prefix_allowed_tokens_fn(1, first_step[1]), [22])
            return torch.cat((first_step, torch.tensor([[3], [22]])), dim=1)

        model = SimpleNamespace(device=torch.device("cpu"), generate=generate)
        self.assertEqual(
            self.model._predict_classification_batch([[8, 9], [8, 9]], model, self.tokenizer),
            ["short", "long"],
        )

    def test_four_bit_load_options_match_compute_and_model_dtypes(self):
        cases = [
            (True, True, "5.24.0", "dtype", torch.bfloat16),
            (True, False, "4.56.0", "dtype", torch.float16),
            (False, False, "4.55.4", "torch_dtype", torch.float32),
        ]
        for cuda, native_bf16, version, dtype_argument, dtype in cases:
            with (
                self.subTest(cuda=cuda, native_bf16=native_bf16, version=version),
                patch("llmabba.llm_abba.torch.cuda.is_available", return_value=cuda),
                patch("llmabba.llm_abba.torch.cuda.is_bf16_supported", return_value=native_bf16) as supports_bf16,
                patch("llmabba.llm_abba.transformers_version", version),
                patch("llmabba.llm_abba.BitsAndBytesConfig") as config,
            ):
                self.assertEqual(_four_bit_load_options(), {
                    dtype_argument: dtype, "quantization_config": config.return_value,
                })
                config.assert_called_once_with(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                )
                if cuda:
                    supports_bf16.assert_called_once_with(including_emulation=False)
                else:
                    supports_bf16.assert_not_called()

    def test_classification_holdout_and_ordered_batch_inference(self):
        features = np.arange(40.).reshape(10, 4)
        labels = np.arange(len(features))
        train, val = self.model.process(
            "example", {"X_data": features, "Y_data": labels}, "classification", "Classify", self.tokenizer,
        )
        train_indices = list(train.with_format("python")["text_outputs"])
        val_indices = list(val.with_format("python")["text_outputs"])
        self.assertTrue(set(train_indices).isdisjoint(val_indices))
        np.testing.assert_allclose(self.model.scaler.mean_, features[train_indices].mean(axis=0))
        self.tokenizer.prompts.clear()
        result = LLMABBA().inference(
            "example", features[:2], "classification", "Classify",
            ToyClassifier([1, 2]), self.tokenizer,
        )
        self.assertEqual(result, ["1", "2"])
        self.assertEqual(len(self.tokenizer.prompts), 2)
        single = LLMABBA().inference(
            "example", features[0], "classification", "Classify",
            ToyClassifier([1]), self.tokenizer,
        )
        self.assertEqual(single, "1")

    def test_regression_uses_training_feature_statistics(self):
        features = pd.DataFrame({"dim": [
            pd.Series([0., 1., 2.]), pd.Series([100., 101., 102.]), pd.Series([1_000.]),
        ]})
        labels = [1., 2., 3.]
        with patch("llmabba.llm_abba.train_test_split",
                   return_value=(features.iloc[:2], features.iloc[2:], labels[:2], labels[2:])):
            train, val = self.model.process(
                "example", {"X_data": features, "Y_data": labels}, "regression", "Predict", self.tokenizer,
            )
        self.assertEqual((len(train), len(val)), (2, 1))
        np.testing.assert_allclose(self.model.scaler.mean_, [51])
        self.assertEqual(self.model.xabba.recap_shape, (2, 3, 1))
        result = LLMABBA().inference(
            "example", features.iloc[2:], "regression", "Predict",
            ToyGenerator(self.tokenizer, ["7.5"]), self.tokenizer,
        )
        self.assertEqual(result, "7.5")
        self.assertEqual(calculate_regression_metrics([1, 2], [1, 3]).at[0, "mae"], 0.5)

    def test_regression_accepts_documented_ndarray(self):
        features = np.arange(24.).reshape(6, 4)
        train, val = self.model.process(
            "array", {"X_data": features, "Y_data": np.arange(6.)}, "regression", "Predict", self.tokenizer,
        )
        self.assertEqual((len(train), len(val)), (4, 2))

    def test_forecasting_with_default_custom_and_numeric_alphabets(self):
        history = np.repeat([[2., 4.]], repeats=30, axis=0)
        for alphabet in (None, ["steady"], -1):
            with self.subTest(alphabet=alphabet):
                train, val = self.model.process(
                    "example", history, "forecasting", "Continue", self.tokenizer,
                    seq_len_pre=3, seq_len_post=2, alphabet_set=alphabet,
                )
                self.assertEqual((len(train), len(val)), (20, 2))
                predicted_symbols = val.with_format("python")[0]["text_outputs"]
                self.assertTrue(predicted_symbols)
                if alphabet is None:
                    self.assertEqual(StringToList(predicted_symbols), ["rise"])
                if alphabet == ["steady"]:
                    self.assertEqual(StringToList(predicted_symbols), ["steady"])
                with self.assertWarns(ShapeWarning):
                    result = LLMABBA().inference(
                        "example", history[:3], "forecasting", "Continue",
                        ToyGenerator(self.tokenizer, [predicted_symbols]), self.tokenizer,
                    )
                np.testing.assert_allclose(result, [[2, 4], [2, 4]])

        self.assertEqual(StringToList(ListToString(["rise", "fall"])), ["rise", "fall"])

    def test_forecast_validation_is_a_chronological_holdout(self):
        data = np.arange(60.).reshape(-1, 2)
        train, val = self.model.process(
            "example", data, "forecasting", "Continue", self.tokenizer, seq_len_pre=3, seq_len_post=2,
        )
        np.testing.assert_allclose(self.model.scaler.mean_, data[:24].mean(axis=0))
        self.assertEqual((len(train), len(val)), (20, 2))
        with self.assertRaisesRegex(ValueError, "separate train and validation"):
            self.model.process(
                "short", data[:9], "forecasting", "Continue", self.tokenizer, seq_len_pre=3, seq_len_post=2,
            )


if __name__ == "__main__":
    unittest.main()
