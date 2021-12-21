"""
This is a script that shows how to convert data.


By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import pipapr

# First we define the path where the data is located
path = '/home/user/folder_containing_data'

# If you don't have any data to try on, you can run the 'example_create_synthetic_dataset.py' script

# We then parse this data using the parser:
tiles = pipapr.tileParser(path=path, frame_size=2048)

# Next, we batch convert the data. Please refer to the original paper to understand and set correctly the
# conversion parameters. The parameters that do not appear below are automatically determined but can also be
# set by the user.
converter = pipapr.tileConverter(tiles)
converter.batch_convert_to_apr(Ip_th=120,               # Intensity thresholding parameter
                               rel_error=0.2,           # Relative error parameter
                               gradient_smoothing=5,    # Gradient smoothing parameters
                               )