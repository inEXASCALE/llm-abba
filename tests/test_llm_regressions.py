import os
from pathlib import Path
from tempfile import TemporaryDirectory
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
from llmabba.llm_abba import ListToString, StringToList
from llmabba.llmabba import LLMABBA as LegacyLLMABBA
from llmabba.utils.regressor_tools import calculate_regression_metrics
from llmabba.xabba import ShapeWarning


class ToyTokenizer:
    all_special_tokens = ("<pad>",)

    def __init__(self):
        self.prompts = []
        self.continuation = ""

    def get_vocab(self):
        return {"rise": 0, "fall": 1, "flat": 2, "<pad>": 3, "▁▁": 4}

    def __call__(self, prompt, max_length=None, return_tensors=None, **kwargs):
        if return_tensors == "pt":
            self.prompts.append(prompt)
            return BatchEncoding({"input_ids": torch.tensor([[8, 9]]), "attention_mask": torch.tensor([[1, 1]])})
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
        result = LLMABBA().inference(
            "example", features[:2], "classification", "Classify",
            ToyGenerator(self.tokenizer, ["first", "second"]), self.tokenizer,
        )
        self.assertEqual(result, ["first", "second"])
        self.assertEqual(len(self.tokenizer.prompts), 2)
        single = LLMABBA().inference(
            "example", features[0], "classification", "Classify",
            ToyGenerator(self.tokenizer, ["first"]), self.tokenizer,
        )
        self.assertEqual(single, "first")

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
