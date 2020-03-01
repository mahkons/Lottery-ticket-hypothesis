from setuptools import setup

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

SHORT_DESCRIPTION = "An implementation of the lottery ticket hypothesis. Reinforcement learning"

DEPENDENCIES = [
    'six',
    'torch',
    'tensorflow',
    'numpy',
    'plotly',
    'tensorboardX',
]

VERSION = "1"
URL = "https://github.com/mahkons/Lottery-ticket-hypothesis"


setup(
    name='lottery-ticket',
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    author='Konstantin Makhnev',
    license='MIT License',

    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],

    keywords='lottery ticket hypothesis',

    packages=['lottery-ticket'],

    install_requires=DEPENDENCIES,
)
