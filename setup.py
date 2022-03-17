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
        'keras~=2.8.0',
        'torch>=1.10.0',
        'pytest~=7.1.0',
        'pytorch-lightning~=1.5.10',
        'torchvision~=0.11.1',
    ]
)
