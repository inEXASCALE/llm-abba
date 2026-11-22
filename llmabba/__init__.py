__version__ = '0.0.6'

from .xabba import (
    ABBA, XABBA, aggregate, compress, fastXABBA, fastXABBA_inc, fastXABBA_len,
    inv_compress, inv_digitize, inv_transform, quantize,
)


def __getattr__(name):
    if name == "LLMABBA":
        from .llm_abba import LLMABBA

        return LLMABBA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
