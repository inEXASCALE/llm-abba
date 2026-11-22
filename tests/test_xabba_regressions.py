import unittest
from unittest.mock import patch

import numpy as np

from llmabba import ABBA, XABBA, compress, fastXABBA, fastXABBA_inc, fastXABBA_len
from llmabba.comp import compress as python_compress
from llmabba.quant import QUANT


class XABBARegressionTests(unittest.TestCase):
    def test_compression_uses_the_same_tolerance_on_both_backends(self):
        generator = np.random.default_rng(42)
        cases = ([0, 0.4, 0], [4], [1, 3, 5, 7], generator.normal(size=30))
        for values in cases:
            with self.subTest(values=values):
                np.testing.assert_allclose(
                    compress(values, tol=0.2, max_len=4),
                    python_compress(values, tol=0.2, max_len=4),
                    atol=1e-12,
                )
        self.assertEqual(compress([0, 0.4, 0], tol=0.2).shape, (2, 4))
        with self.assertRaises(ValueError):
            compress([0, np.nan, 1])

    def test_fast_encoders_accept_integer_and_single_point_inputs(self):
        for cls in (fastXABBA, fastXABBA_len, fastXABBA_inc):
            with self.subTest(encoder=cls):
                encoder = cls(k=1, verbose=False)
                self.assertEqual(encoder.inverse_transform(encoder.transform([3]), start=3), [3])
                np.testing.assert_allclose(
                    encoder.inverse_transform(encoder.transform([0, 1, 2]), start=0), [0, 1, 2],
                )

    def test_partitioning_keeps_every_sample(self):
        values = np.arange(10.0)
        model = XABBA(partition=3, verbose=0)
        symbols, starts = model.fit_transform(values, n_jobs=1, return_start_set=True)
        np.testing.assert_allclose(model.inverse_transform(symbols, starts), values, atol=1e-3)
        transformed, starts = model.transform(values, n_jobs=3)
        self.assertEqual(starts, [0, 4, 7])
        np.testing.assert_allclose(model.inverse_transform(transformed, starts), values, atol=1e-3)

    def test_default_parallelism_handles_short_sequences(self):
        with patch("llmabba.xabba.get_cpu_affinity", return_value=range(64)):
            values = [0, 1, 2]
            model = ABBA(verbose=0)
            np.testing.assert_allclose(model.decode(model.encode(values)), values)
            symbols, starts = model.transform(values)
            np.testing.assert_allclose(model.inverse_transform(symbols, starts), values)

    def test_transform_uses_the_training_distance(self):
        rows = np.array([np.linspace(0, 100, 21), np.resize([0, 200], 21)])
        model = XABBA(init="kmeans", k=3, scl=3, tol=0.01, verbose=0).fit(rows, n_jobs=1)
        centers = np.column_stack((
            model.quantizer_len.dequant(model.parameters.centers[:, 0]),
            model.quantizer_inc.dequant(model.parameters.centers[:, 1]),
        ))
        piece = [1, 99]
        raw = np.argmin(np.linalg.norm(centers - piece, axis=1))
        weighted = np.argmin(np.linalg.norm((centers - piece) * [model.scl, 1] / model._std, axis=1))
        self.assertNotEqual(raw, weighted)
        symbols, _ = model.transform([[0, 99]], n_jobs=1)
        self.assertEqual(symbols, [[model.parameters.alphabets[weighted]]])

    def test_missing_values_and_constant_centers_roundtrip(self):
        for method, values, expected in [
            ("ffill", [[np.nan, 1, 2]], [0, 1, 2]),
            ("bfill", [[2, 1, np.nan]], [2, 1, 0]),
            ("mean", [[np.nan, np.nan]], [0, 0]),
        ]:
            with self.subTest(method=method):
                model = XABBA(fillna=method, verbose=0)
                symbols = model.fit_transform(values, n_jobs=1)
                np.testing.assert_allclose(model.inverse_transform(symbols), [expected])
                transformed, starts = model.transform(values, n_jobs=1)
                np.testing.assert_allclose(model.inverse_transform(transformed, starts), [expected])
        quantizer = QUANT()
        np.testing.assert_allclose(quantizer.dequant(quantizer(np.array([3., 3.]))), [3, 3])

    def test_recast_does_not_mutate_input(self):
        rows = [[1, 2, 3, 4]]
        model = XABBA()
        for _ in range(2):
            self.assertEqual(model.recast_shape(rows, recap_shape=(1, 2, 2)).shape, (1, 2, 2))
        self.assertEqual(rows, [[1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
