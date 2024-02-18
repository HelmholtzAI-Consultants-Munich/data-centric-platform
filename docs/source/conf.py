# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DCP'
copyright = '2024, Christina Bukas, Mariia Koren, Helena Pelin'
author = 'Christina Bukas, Mariia Koren, Helena Pelin'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys
from pathlib import Path
import sphinx_rtd_theme

# Add parent dir to known paths
p = Path(__file__).parents[2]
sys.path.insert(0, os.path.abspath(p))

# Add the following extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
]

# Use RTD theme
html_theme = "sphinx_rtd_theme"
