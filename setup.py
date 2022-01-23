import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micplot",
    version="0.1.0",
    author="Sjoerd Cornelissen",
    description="Effective visualization with one line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SjoerdCor/micplot",
    packages=setuptools.find_packages(),
    install_requires=["pandas==1.3", "matplotlib==3.5", "numpy==1.21"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
