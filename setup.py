from setuptools import setup, find_packages

setup(
    name='BoneEnhance',
    version='0.1',
    author='Santeri Rytky',
    author_email='santeri.rytky@oulu.fi',
    packages=find_packages(),
    include_package_data=False,
    license='LICENSE',
    long_description=open('README.md').read(),
)