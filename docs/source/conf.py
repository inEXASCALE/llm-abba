import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llmabba import __version__ as release


project = 'llmabba'
copyright = 'Erin Carson, Xinye Chen, and Cheng Kang'
author = 'InEXASCALE computing'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]
templates_path = ['_templates']
source_suffix = '.rst'
exclude_patterns = []
locale_dirs = ['locale/']
gettext_compact = False

exclude_patterns = []

pygments_style = 'lovelace'

html_theme = "sphinx_rtd_theme" # html_theme = 'alabaster'
html_theme_options = {
    'logo_only': False,
    'navigation_depth': 5,
}
html_static_path = []
