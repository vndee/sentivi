import os
import sys

project = 'Sentivi'
copyright = '2020, Duy V. Huynh'
author = 'Duy V. Huynh'
version = '1.0.4'
release = '1.0.4'
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['static']

autodoc_default_options = {
    'special-members': '__init__',
}

sys.path.insert(0, os.path.abspath('../..'))
extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme']
