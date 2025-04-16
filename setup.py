from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="COMPASS",
    version="0.1.0",
    author="Berkay Günes",
    description="COMPASS: A Python package for comparing bayesian models in a simulation based setting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "compass"},
    packages=find_packages(where="compass"),
    url="https://github.com/bGuenes/COMPASS",
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "torch",
        "schedulefree"
    ],)