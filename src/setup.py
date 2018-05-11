#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

description = ('Replicating Latent State Approaches to Social Network'
               ' Analysis (Hoff, Raftery, Handcock 2002)')

with open('../README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'funcy',
    'ipykernel',
    'matplotlib',
    'numpy',
    'pandas',
    'requests',
    'scipy',
    'seaborn',
    'tqdm',
]

setup(
    author="Micah Smith",
    author_email='micahs@mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description=description,
    install_requires=requirements,
    license='MIT license',
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    name='horaha',
    packages=find_packages(include=['semsch', 'horaha', 'horaha.*']),
    python_requires='>=3.5',
    version='0.1',
    zip_safe=False,
)
