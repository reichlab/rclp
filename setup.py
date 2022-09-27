#!/usr/bin/env python

from setuptools import setup

setup(
    name='rclp',
    version='1.0.0',
    description='Re-calibrated linear pools',
    author='Serena Wang, Evan L. Ray',
    author_email='elray@umass.edu',
    url='https://github.com/reichlab/rclp',
    py_modules=['rclp'],
    install_requires=[
        'numpy',
        'tensorflow>=2',
        'tensorflow_probability>=0.16.0'
    ]
)
