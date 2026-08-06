"""Sphinx configuration for the ChebyshevND documentation."""

import shutil
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "ChebyshevND"
author = "Philip Lynch"
copyright = "2026, Philip Lynch"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_member_order = "bysource"

# Render the example notebooks from their stored outputs; don't re-execute
# during the docs build.
nb_execution_mode = "off"

# The example notebooks live at the repository root; copy them into the docs
# source tree so the toctree can reference them.
_docs = Path(__file__).parent
_examples_src = _docs.parent / "examples"
_examples_dst = _docs / "examples"
_examples_dst.mkdir(exist_ok=True)
for _nb in _examples_src.glob("*.ipynb"):
    shutil.copy(_nb, _examples_dst / _nb.name)

templates_path = []
exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "ChebyshevND"
