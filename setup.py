from distutils.core import setup

setup(
    name="dccm_quasistatic",
    version="1.0.0",
    packages=["dccm_quasistatic"],
    install_requires=[
        "numpy",
        "matplotlib",
        "ipywidgets",
        "pre-commit",
        "tqdm",
        # "manipulation",
        # "underactuated",
        "black",
        # Quasistatic Simulator Requirements
        "parse",
        "diffcp",
        "cvxpy",
        "meshcat",
        "pytest",
    ],
)

# Drake built from source, and quasistatic sim built from source
