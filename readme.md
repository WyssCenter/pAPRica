# How to install

- Download pyapr from github
- Compile it with ``EXTRA_CMAKE_ARGS="-DPYAPR_USE_OPENMP=ON -DPYAPR_USE_CUDA=OFF" python setup.py install
``
- Create a dedicated python env
- Activate new env
- Install pipapr dependencies using environment.yml
- Enjoy

# How to use