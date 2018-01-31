#!/bin/bash

# Tox does not work well with anaconda
# First, remove anaconda from the path so that other version python will be used
export PATH=`echo ${PATH} | awk -v RS=: -v ORS=: '/anaconda/ {next} {print}'`

# Python 2.7 should have been installed
# Otherwise download it here https://www.python.org/downloads/release/python-279
python2.7 --version

# Python 3.6 should have been installed
# Otherwise download it here https://www.python.org/downloads/release/python-352
python3.5 --version

tox
