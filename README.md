# LLM-ABBA

[![PyPI version][pypi-badge]][pypi] [![Documentation build][docs-badge]][docs] [![arXiv paper][paper-badge]][paper] [![Cython acceleration][cython-badge]][getting-started] [![MIT license][license-badge]][license]

LLM-ABBA converts numerical time series into compact symbol sequences. Use `ABBA` to encode and reconstruct signals, or use `LLMABBA` to fine-tune a language model for classification, regression, or forecasting.

## Installation

```bash
python -m pip install llmabba
```

See the [installation guide][getting-started] for source installation and GPU setup.

## Quick start

### Encode and reconstruct a time series

```python
from llmabba import ABBA

series = [[1.2, 1.4, 1.3, 1.8, 2.2, 2.4, 2.1]]
abba = ABBA(tol=0.1, alpha=0.1, verbose=0)
symbols = abba.encode(series)
reconstruction = abba.decode(symbols)
```

### Classify ECG signals with Mistral-7B

After [installing from source][getting-started], run the PTBDB demo from the repository root on a CUDA setup that supports bitsandbytes 4-bit quantization.

```bash
cd examples
WANDB_MODE=disabled python demo-ts-classification.py --model mistralai/Mistral-7B-Instruct-v0.1
```

## Documentation

- [Getting started][getting-started] covers installation, CUDA setup, and symbolic approximation.
- [PTBDB classification tutorial][classification] covers data splits, class balancing, and evaluation on the held-out test set.
- [Regression and forecasting][workflows] covers input shapes, generation, and checkpoint reuse.
- [API reference and citations][docs] are available in the full documentation.

## Contributing and license

Report bugs or ask questions in [GitHub Issues][issues]. Documentation and code contributions are welcome. LLM-ABBA is released under the [MIT License][license]. For research use, see the [citation details][citation].

[pypi-badge]: https://img.shields.io/pypi/v/llmabba?style=flat-square&logo=pypi&logoColor=white&label=PyPI&color=2563eb
[docs-badge]: https://img.shields.io/readthedocs/llm-abba/latest?style=flat-square&logo=readthedocs&logoColor=white&label=Docs
[paper-badge]: https://img.shields.io/badge/arXiv-2411.18506-b31b1b?style=flat-square&logo=arxiv&logoColor=white
[cython-badge]: https://img.shields.io/badge/Cython-accelerated-0f766e?style=flat-square
[license-badge]: https://img.shields.io/github/license/inEXASCALE/llm-abba?style=flat-square&color=64748b
[pypi]: https://pypi.org/project/llmabba/
[docs]: https://llm-abba.readthedocs.io/en/latest/
[paper]: https://arxiv.org/abs/2411.18506
[license]: https://github.com/inEXASCALE/llm-abba/blob/master/LICENSE
[getting-started]: https://github.com/inEXASCALE/llm-abba/blob/master/docs/source/getting-started.rst
[classification]: https://github.com/inEXASCALE/llm-abba/blob/master/docs/source/classification.rst
[workflows]: https://github.com/inEXASCALE/llm-abba/blob/master/docs/source/workflows.rst
[citation]: https://github.com/inEXASCALE/llm-abba/blob/master/docs/source/references.rst
[issues]: https://github.com/inEXASCALE/llm-abba/issues
