from setuptools import setup, find_packages

setup(
    name='ptmpc',
    version='0.0.1',
    url='https://github.com/zanellia/ptmpc.git',
    author='Andrea Zanelli',
    author_email='znllandrea@gmail.com',
    description='Python implementation of partially tightened MPC',
    packages=find_packages(),    
    install_requires=['numpy', 'casadi', 'matplotlib', 'PyQt5'],
)
