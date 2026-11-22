Artifacts and symbol helpers
============================

.. py:currentmodule:: llmabba.llm_abba

The names below are also importable from ``llmabba.llmabba`` for compatibility. :class:`llmabba.LLMABBA` is available from both module paths and from ``llmabba``.

.. autofunction:: save_abba

   Write a fitted ABBA encoder, scaler, or other pickle-compatible object to ``filename``. The parent directory must exist. The return value is ``None``.

   .. testcode:: artifacts

      from pathlib import Path
      from tempfile import TemporaryDirectory

      from llmabba import ABBA
      from llmabba.llm_abba import save_abba, load_abba

      encoder = ABBA(verbose=0)
      symbols = encoder.encode([[0.0, 1.0, 2.0]])
      with TemporaryDirectory() as directory:
          filename = Path(directory) / "encoder.pkl"
          save_abba(encoder, filename)
          restored = load_abba(filename)
      assert restored.decode(symbols) == encoder.decode(symbols)

.. autofunction:: load_abba

   Return the object saved at ``filename``. The encoder save-and-load example above uses the restored codebook after the file is closed. Only open pickle artifacts from trusted sources.

.. autofunction:: ListToString

   Join an iterable of string symbols with one space. Returns a string. Use symbols without whitespace if you plan to split the sequence again.

   .. testcode:: helpers

      from llmabba.llm_abba import ListToString, StringToList

      text = ListToString(["rise", "fall"])
      assert text == "rise fall"

.. autofunction:: StringToList

   Split ``text`` on runs of whitespace and return a list of symbols.

   .. testcode:: helpers

      assert StringToList(text) == ["rise", "fall"]

.. autofunction:: cross_correlation

   Return the normalized, zero-lag correlation of two aligned numeric sequences. Use equal-length nonempty inputs with nonzero variance.

   .. testcode::

      import math
      from llmabba.llm_abba import cross_correlation

      assert math.isclose(cross_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
