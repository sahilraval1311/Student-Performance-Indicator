from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Reads a requirements file and returns a list of requirements.
    """
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="Student Performance Prediction",
    author="Sahil Raval",
    author_email="ravalsahil1311@gmail.com",
    packeges=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
