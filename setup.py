from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Implementation of the GraphReader Paper'
LONG_DESCRIPTION = 'A package that allows to build a RAG Agent using a Graph Knowledge base.'

# Setting up
setup(
    name="graphreader",
    version=VERSION,
    author="Atharva Bagde, Sushant Menon",
    author_email="sushantmenon1@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
    "langchain==0.3.4",
    "langchain-openai==0.2.3",
    "networkx==3.4.1",
    "nltk==3.9.1",
    "openai==1.52.0",
    "pinecone-client==5.0.1",
    "pypdf==5.0.1",
    "sentence-transformers==3.2.0",
    "tqdm==4.66.5"
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)