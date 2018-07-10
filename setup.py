from setuptools import setup, find_packages, Extension

setup(
    name="dlab",
    packages=['dlab'],
    version="0.0.1",
    description="Python code for Dwivedi Lab",
    author='Janahan Selvanayagam',
    author_email='seljanahan@hotmail.com',
    url='https://github.com/dwivedilab/dlab',
    keywords=['EEG'],
    install_requires=[],
    include_package_data=True,
    zip_safe=False
)

#entry_points={'console_scripts': ['pyinstrument = pyinstrument.__main__:main']},
