# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FeaSel-Net'
copyright = '2022, Felix Fischer'
author = 'Felix Fischer'
release = '0.0.8'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon']

source_suffix = {'.rst': 'restructuredtext',
                 '.txt': 'restructuredtext',
                 '.md': 'markdown',
                 }

autodoc_default_options = {
    'show-inheritance': True,
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'autodoc-inherit-docstrings': False,
}

autodoc_inherit_docstrings = False

autoclass_content = 'both'

templates_path = ['_templates']
exclude_patterns = ['_build',
                    'Thumbs.db',
                    '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
	"github_url": "https://github.tik.uni-stuttgart.de/FelixFischer/FeaSel-Net",
	}
html_context = {
   "default_mode": "light"
}

html_sidebars = {
  "pagename": ['information/SOE']
}
html_theme_options = {
  "show_nav_level": 2
}
