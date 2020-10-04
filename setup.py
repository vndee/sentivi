import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()
requires = (HERE / 'requirements.txt').read_text().split('\n')

setup(
    name="sentivi",
    version="1.0.8",
    description="A simple tool for Vietnamese Sentiment Analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/vndee/sentivi",
    author="Duy V. Huynh",
    author_email="hvd.huynhduy@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=('data', 'test')),
    include_package_data=True,
    install_requires=requires
)
