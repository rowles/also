#!/usr/bin/env python

from distutils.core import setup

setup(
    name="also",
    version="1.0",
    description="Attribute-wise Learning for Scoring Outliers",
    author="Andrew Rowles",
    author_email="andrew@rowles.io",
    url="https://github.com/rowles/also",
    packages=["also"],
    install_requires=["numpy", "scikit-learn"],
)
