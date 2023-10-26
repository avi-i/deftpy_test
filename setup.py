from setuptools import setup, find_packages

setup(
    name='deftpy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'ase',
        'matplotlib',
        'pymatgen',
        'pymatgen-analysis-defects',
        'tqdm',
    ],
)