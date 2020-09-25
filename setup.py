import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(
    name="sentivi",
    version="0.0.1",
    description="A simple tool for Vietnamese Sentiment Analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/vndee/deepgen",
    author="Duy Huynh",
    author_email="hvd.huynhduy@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=('example', 'test')),
    include_package_data=True,
    install_requires=['torch==1.1.0',
                      'torchvision==0.3.0',
                      'tqdm==4.32.2'],
)