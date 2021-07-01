"""

By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import subprocess
import os

for n_cores in [1, 2, 4, 8, 16, 24, 32, 48]:
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    proc = subprocess.run('python ./benchmark_max_proj.py', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(proc.stdout)