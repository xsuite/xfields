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
    version='0.5.0',
    description='Field Maps and Particle In Cell',
    url='https://github.com/xsuite/xfields',
    author='Giovanni Iadarola',
    packages=find_packages(),
    ext_modules = extensions,
    include_package_data=True,
    install_requires=[
        'numpy>=1.0',
        'scipy',
        'pandas',
        'xobjects>=0.0.4',
        'xtrack>=0.0.1',
        ]
    )
