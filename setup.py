from setuptools import setup, find_packages, Extension

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []


#########
# Setup #
#########

setup(
    name='xfields',
    version='0.1.0',
    description='Field Maps and Particle In Cell',
    url='https://github.com/xsuite/xfields',
    author='Giovanni Iadarola',
    packages=find_packages(),
    ext_modules = extensions,
    install_requires=[
        'numpy>=1.0',
        'scipy',
        'xobjects>=0.0.4',
        'xtrack>=0.0.1',
        ]
    )
