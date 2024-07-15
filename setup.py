from setuptools import setup, find_packages

__version__ = "0.2.1"

packages = find_packages(
    exclude=[
        "tests",
        "examples",
    ]
)

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="interpreter",
    version=__version__,
    license="MIT",
    packages=packages,
    include_package_data=True,
    python_requires=">=3.9",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hector Kohler",
    author_email="hector.kohler@inria.fr",
    install_requires=[
        "scikit-learn>=1.3.0",
        "stable-baselines3",
        "rlberry",
        "gymnasium[mujoco]",
        "huggingface-sb3",
        "tqdm",
        "gym",
    ],
)
