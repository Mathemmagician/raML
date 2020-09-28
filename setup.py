
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="raML", # Replace with your own username
    version="0.0.0",
    author="Ramil Aleskerov",
    author_email="ramilraleskerov@gmail.com",
    description="The Best Machine Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mathemmagician/raML",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)