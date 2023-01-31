Generating documentation
========================

The documentation is built using ``sphinx``. It can be generated locally by the following steps:
::
    pip install sphinx sphinx-rtd-theme myst_nb
    cd doc
    make html

If using Windows, change the ``make html`` command to ``make.bat html``.