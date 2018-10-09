from setuptools import setup, find_packages, Extension

setup(
    name="dlab",
    packages=['dlab'],
    version="1.0.2",
    description="Python code for Dwivedi Lab",
    author='Janahan Selvanayagam',
    author_email='seljanahan@hotmail.com',
    url='https://github.com/dwivedilab/dlab',
    keywords=['EEG'],
    install_requires=[
        'pandas',
        'savReaderWriter',
        'matplotlib',
        'numpy',
        'xlrd'
    ],
    include_package_data=True,
    zip_safe=False
)
