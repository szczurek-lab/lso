from setuptools import find_packages
from setuptools import setup

setup(
    name='LSO',
    version='0.1',
    description='Python package for a latent space optimization.',
    author='Marcin Możejko & Adam Izdebski',
    author_email='mmozejko1988@gmail.com',
    url='https://github.com/szczurek-lab/lso',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
    ]
)
