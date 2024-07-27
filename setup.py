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


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)
        
        
setup_args = {'name':"xabba",
        'packages':setuptools.find_packages(),
        'version':"0.0.1",
        'cmdclass': {'build_ext': CustomBuildExtCommand},
        'install_requires':["numpy>=1.3.0", "scipy>=0.7.0", 
                            "requests", "pandas", 
                            "scikit-learn", "pychop", "cython>=0.27",
                            "joblib>=1.1.1",
                            "matplotlib"],
        'packages':{"xabba"},
        'package_data':{"xabba": ["xabba"]},
        'long_description':long_description,
        'author':"NA",
        'author_email':"noname@email.com",
        'classifiers':["Intended Audience :: Science/Research",
                    "Intended Audience :: Developers",
                    "Programming Language :: Python",
                    "Topic :: Software Development",
                    "Topic :: Scientific/Engineering",
                    "Operating System :: Microsoft :: Windows",
                    "Operating System :: Unix",
                    "Programming Language :: Python :: 3"
                    ],
        'description':"",
        'long_description_content_type':'text/x-rst',
        'url':"https://github.com/NA",
        'license':'BSD 3-Clause'
    }


comp = Extension('xabba.comp',
                        sources=['xabba/comp.pyx'])

agg = Extension('xabba.agg',
                        sources=['xabba/agg.pyx'])

inverse= Extension('xabba.inverse',
                        sources=['xabba/inverse.pyx'])


try:
    setuptools.setup(
        setup_requires=["cython", "numpy>=1.17.3"],
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
    setuptools.setup(setup_requires=["numpy>=1.17.3"], **setup_args)
    