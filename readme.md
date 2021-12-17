# How to install

- Download from the repo
- Cd into the folder
- run ``python setup.py install`` or ``python setup.py develop``

# How to generate documentation

The documentation can be automatically generated using pdoc. First install pdoc:

``pip install pdoc3``

Then you can generate the doc:

``pdoc3 --html --output-dir doc pipapr --force``