from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "nlpoptnet" / "src"

sys.path.insert(0, str(SRC))

project = "NLPOptNet"
copyright = "2026, NLPOptNet Authors"
author = "Bimol Nath Roy, Rahul Golder, MM Faruque Hasan"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "NLPOptNet Documentation"
# html_static_path = []
html_static_path = ['_static']
html_css_files = ['custom.css']

source_suffix = ".rst"

html_context = {
    "display_github": True,
    "github_user": "SOULS-TAMU",
    "github_repo": "NLPOpt-Net",
    "github_version": "main",  # or "master"
    "conf_py_path": "/docs/",
}

latex_elements = {
    "extraclassoptions": "openany,oneside",
}