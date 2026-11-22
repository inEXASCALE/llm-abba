import logging
import setuptools
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from setuptools import Extension
import warnings

try:
    from Cython.Distutils import build_ext
except ImportError as e:
    warnings.warn(f"Cython not found: {e}")
    from setuptools.command.build_ext import build_ext

try:
    with open("README.md", 'r', encoding='utf-8') as f:
        long_description = f.read()
except Exception as e:
    long_description = ""
    warnings.warn(f"Failed to read README.md: {e}")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)
ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, SystemExit)

def get_version(fname):
    version = '0.0.2'
    package = 'llmabba'
    try:
        with open(fname) as f:
            for line in f:
                if line.startswith("__version__ = '"):
                    version = line.split("'")[1]
                elif line.startswith("__name__ = '"):
                    package = line.split("'")[1]
    except Exception as e:
        log.error(f"Failed to parse version from {fname}: {e}")
    return version, package

__version__, __package__ = get_version('llmabba/__init__.py')

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        try:
            import numpy
            self.include_dirs.append(numpy.get_include())
            log.info("NumPy include directory added: %s", numpy.get_include())
        except ImportError as e:
            log.error(f"NumPy not found: {e}")
            raise
        build_ext.run(self)

setup_args = {
    'name': __package__,
    'packages': setuptools.find_packages(),
    'version': __version__,
    'cmdclass': {'build_ext': CustomBuildExtCommand},
    'install_requires': [
        "tf_keras",
        "wandb",
        "numpy>=1.20",
        "scipy>=0.7.0",
        "requests",
        "pandas",
        "scikit-learn",
        "joblib>=1.1.1",
        "packaging",
        "torch",
        "transformers",
        "bitsandbytes",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "matplotlib"
    ],
    'package_data': {__package__: ['*.pyx', '*.pxd', '*.c', '*.pyd']},
    'long_description': long_description,
    'author': "Erin Carson, Xinye Chen, Cheng Kang",
    'author_email': "xinyechenai@email.com",
    'classifiers': [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3"
    ],
    'description': "LLM-ABBA: mining time series via symbolic approximation and large language models",
    'long_description_content_type': 'text/markdown',
    'url': "https://github.com/inEXASCALE/llm-abba",
    'license': 'MIT'
}

# Define Cython extensions
comp = Extension('llmabba.compfp', sources=['llmabba/compfp.pyx'])
agg = Extension('llmabba.aggfp', sources=['llmabba/aggfp.pyx'])
inverse = Extension('llmabba.inversefp', sources=['llmabba/inversefp.pyx'])

try:
    log.info("Attempting to build with Cython extensions")
    setuptools.setup(
        **setup_args,
        ext_modules=[comp, agg, inverse]
    )
except ext_errors as ext_reason:
    log.error(f"C extension compilation failed: {ext_reason}")
    log.warning("Falling back to setup without C extensions")
    if 'build_ext' in setup_args['cmdclass']:
        del setup_args['cmdclass']['build_ext']
    setuptools.setup(**setup_args)
