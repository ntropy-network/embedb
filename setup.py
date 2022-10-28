import os

from setuptools import find_packages, setup
from embedb import __version__

# Package meta-data.
NAME = "embedb"
DESCRIPTION = "EmbeDB is a small Python wrapper around LMDB built as key-value storage for embeddings."
URL = "https://github.com/ntropy-network/embedb"
REQUIRES_PYTHON = ">=3.7.0"
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        lineiter = f.read().splitlines()
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name=NAME,
    version=__version__,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=[
        "Machine Learning",
        "LMDB",
        "Embeddings",
        "Databases",
    ],
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt"),
    include_package_data=True,
)