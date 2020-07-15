from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gauge_core", 
    version='0.1', 
    author="Mihailo Isakov",
    author_email="isakov.m@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',     
        'pandas',
        'networkx',
        'hdbscan'
    ]
)
