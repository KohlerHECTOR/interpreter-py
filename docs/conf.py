# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "interpreter"
copyright = "2024, Hector Kohler"
author = "Hector Kohler"
release = "0.1.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    # "sphinx_gallery.gen_gallery",
    # "sphinx_design",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
    "numpydoc",
]

myst_enable_extensions = ["dollarmath", "amsmath"]

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# generate autosummary even if no references
autosummary_generate = True
autodoc_inherit_docstrings = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme_options = {
    "header_links_before_dropdown": 4,
    "logo": {
        "text": "Interpreter",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/KohlerHECTOR/interpreter-py",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": False,
    "show_toc_level": 1,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    # "show_nav_level": 2,
    "footer_start": ["copyright"],
    "secondary_sidebar_items": [],
    "navbar_persistent": ["search-button"],
    "navbar_center": ["navbar-nav"],
}

# html_sidebars = {
#     "community/index": [
#         "sidebar-nav-bs",
#         "custom-template",
#     ],  # This ensures we test for custom sidebars
# }

html_sidebars = {"**": ["page-toc"]}

html_context = {
    "github_user": "KohlerHECTOR",
    "github_repo": "interpreter-py",
    "github_version": "main",
    "doc_path": "docs",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
