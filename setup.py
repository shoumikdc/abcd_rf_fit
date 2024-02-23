"""
Setup
"""

import os

from setuptools import setup, find_namespace_packages

# TODO: update with your package name and details
DIST_NAME = "abcd_rf_fit"  # some people prefer using example-package instead as the distribution name
PACKAGE_NAME = "abcd_rf_fit"

REQUIREMENTS = [
    "scipy",
    "numpy",
    "sympy",
    "matplotlib",
]

# We strongly suggest using the packages listed below.
EXTRA_REQUIREMENTS = {
    "dev": [
        "jupyterlab>=3.1.0",
    ],
}

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), PACKAGE_NAME, "VERSION.txt")
)

with open(version_path, "r") as fd:
    version_str = fd.read().rstrip()

# TODO: update with your project details
setup(
    name=DIST_NAME,
    version=version_str,
    description=DIST_NAME,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/shoumikdc/abcd_rf_fit",
    author="Shoumik Chowdhury",
    author_email="shoumikc@mit.edu",
    license="Apache",
    packages=find_namespace_packages(exclude=["tutorials*"]),
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[],
    keywords="abcd_rf_fit",
    python_requires=">=3.7",
    project_urls={  # TODO: update with your project urls
        "Documentation": "https://github.com/shoumikdc/abcd_rf_fit",
        "Source Code": "https://github.com/shoumikdc/abcd_rf_fit",
    },
    include_package_data=True,
)
