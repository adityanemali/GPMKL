from setuptools import setup, find_packages

setup(
    name='GPMKL',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    author='Aditya Nemali',
    author_email='aditya.nemali@dzne.de',
    description='Gaussian Process-based prediction of memory performance and biomarker status in ageing and Alzheimer’s disease – A systematic model evaluation',
    url='https://www.sciencedirect.com/science/article/pii/S1361841523001731?via%3Dihub',
)