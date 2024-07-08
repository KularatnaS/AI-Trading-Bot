from setuptools import setup, find_packages

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="llm",
    version="0.0.1",
    description="llm",
    url="https://shash",
    author="Shashitha Kularatna",
    author_email="shashitha_kula@hotmail.com",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.8.*'
)