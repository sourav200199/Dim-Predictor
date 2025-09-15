'''
    This file is used to set up the package for distribution.
'''

from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r') as file:
            requirements = file.readlines()
            requirements = [req.replace("\n","") for req in requirements if req.strip() and not req.startswith('-')]

        return requirements
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []


setup(
    name = 'dim-predictor',
    version = '0.0.1',
    author = 'Sourav',
    author_email = 'sourav.992001@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'),
)