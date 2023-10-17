from setuptools import setup, find_packages

setup(
    name='imzml2np',
    version='1.0',
    description='Extract ion images from imzml to numpy arrays',
    url='https://gitlab.com/Buglakova/sc_isotope_tracing',
    packages=find_packages(include=['sciso']),
    python_requires='>=3.6',
    install_requires=[],
    author='Elena Buglakova',
    author_email='elena.buglakova@embl.de',
    license='MIT'
)