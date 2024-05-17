"""Python setup.py for project_name package"""
import io
import os
from setuptools import find_packages, setup

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("research_roc_utils", "0.1.1")
    '0.1.1'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

VERSION = '0.1.1'

setup(
    name="research_roc_utils",
    version=VERSION,
    description="Utility functions to assist in the computation of ROC curve comparisons based on academic research.",
    url="https://github.com/jpbruehw/research-roc-utils",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="J.P. Bruehwiler",
    packages=find_packages(exclude=["local-dev", ".github"]),
    install_requires=read_requirements("requirements.txt"),
)