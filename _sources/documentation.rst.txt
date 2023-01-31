Generating documentation
========================

The documentation is built using ``sphinx``. It can be generated locally by executing the following steps from the ``doc`` directory:
::
    pip install sphinx sphinx-rtd-theme myst_nb
    make html

.. note:: If using Windows, change the ``make html`` command to ``make.bat html``.
