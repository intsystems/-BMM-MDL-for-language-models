from distutils.core import setup
from setuptools import find_packages
import sys


if sys.version_info < (3, 8):
    exit_string = (
        'Sorry, Python < 3.8 is not supported. '
        f'Current verion is {".".join(map(str, sys.version_info[:2]))}'
    )
    sys.exit(exit_string)

setup(
    name="problib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn"
    ],
)