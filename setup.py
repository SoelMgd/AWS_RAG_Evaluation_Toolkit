from setuptools import setup, find_packages

setup(
    name='AWS_RAG_Toolkit',
    version='0.1',
    packages=find_packages(include=['Core', 'Core.*']),
    include_package_data=True,
    install_requires=[
        # Liste de tes dépendances
    ],
)