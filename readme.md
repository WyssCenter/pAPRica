# How to install

- Download pyapr from github
- Compile it with ``EXTRA_CMAKE_ARGS="-DPYAPR_USE_OPENMP=ON -DPYAPR_USE_CUDA=OFF" python setup.py install
``
- Create a dedicated python env
- Activate new env
- Install pipapr dependencies using environment.yml
- Enjoy

# How to generate documentation

The documentation can be automatically generated using pdoc. First install pdoc:

``pip install pdoc3``

Then you can generate the doc:

``pdoc3 --html --output-dir doc pipapr --force``