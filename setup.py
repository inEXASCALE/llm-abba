import logging
import setuptools
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from setuptools import Extension
import warnings

try:
    from Cython.Distutils import build_ext
except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools.command.build_ext import build_ext

with open("README.rst", 'r') as f:
    long_description = f.read()

logging.basicConfig()
log = logging.getLogger(__file__)
ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, SystemExit)


def get_version(fname):
    version = '0.0.1'
    package = 'llmabba'

    with open(fname) as f:
        for line in f:
            if line.startswith("__version__ = '"):
                version = line.split("'")[1]

            elif line.startswith("__name__ = '"):
                package = line.split("'")[1]

    return version, package
    # raise RuntimeError('Error in parsing version string.')


__version__, __package__ = get_version('llmabba/__init__.py')


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


setup_args = {'name': __package__,
              'packages': setuptools.find_packages(),
              'version': __version__,
              'cmdclass': {'build_ext': CustomBuildExtCommand},
              'setup_requires': ["cython", "numpy>=1.17.3"],
              'install_requires': ["numpy>=1.3.0",
                                   "scipy>=0.7.0",
                                   "requests",
                                   "pandas",
                                   "scikit-learn",
                                   "pychop",
                                   "cython>=0.27",
                                   "joblib>=1.1.1",
                                   "transformers",
                                   "peft",
                                   "trl",
                                   "datasets",
                                   "accelerate",
                                   "matplotlib"],

              'package_data': {__package__: [__package__]},
              'long_description': long_description,
              'author': "NA",
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
              'long_description_content_type': 'text/x-rst',
              'url': "https://github.com/inEXASCALE/llmabba",
              'license': 'BSD 3-Clause'
              }

comp = Extension('llmabba.comp',
                 sources=[__package__ + '/comp.pyx'])

agg = Extension('llmabba.agg',
                sources=[__package__  + '/agg.pyx'])

inverse = Extension('llmabba.inverse',
                    sources=[__package__  + '/inverse.pyx'])

try:
    setuptools.setup(
        **setup_args,
        ext_modules=[comp,
                     agg,
                     inverse
                     ],
    )

except ext_errors as ext_reason:
    log.warning(ext_reason)
    log.warning("The C extension could not be compiled.")
    if 'build_ext' in setup_args['cmdclass']:
        del setup_args['cmdclass']['build_ext']
    setuptools.setup(**setup_args)
