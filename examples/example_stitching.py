"""
This is a script that shows how to stitch data.


By using this code you agree to the terms of the software license agreement.

© Copyright 2020 Wyss Center for Bio and Neuro Engineering – All rights reserved
"""

import paprica

# First we define the path where the data is located
path = '/home/user/folder_containing_data'

# If you don't have any data to try on, you can run the 'example_create_synthetic_dataset.py' script

# We then parse this data using the parser
tiles = paprica.tileParser(path=path, frame_size=2048, ftype='apr')
# The previous parser expect the data to be in y_x.apr naming convention, this behavior can be adapted by creating
# your own parsing class that inherits from tileParser. We provide an example for COLM data where we just need to
# parse the data to associate each tile with its position on the grid:
tiles_autofluo = paprica.parser.colmParser(path=path, channel=0)
tiles_nissl = paprica.parser.colmParser(path=path, channel=1)
# Each channel is parsed independently to give maximum freedom for stitching and display.

# We can then use the stitcher to stitch this dataset, we just need to give it the tiles and the expected overlaps.
# There are no limitation in the overlaps amount (it can be more than 50%), but for the higher chances of success it
# should be close to the real overlaps.
stitcher_expected = paprica.tileStitcher(tiles_nissl, overlap_h=20, overlap_v=20)

# This will arrange the tile with the expected overlap and can then be used to display the dataset quickly
stitcher_expected.compute_expected_registration()
stitcher_expected.reconstruct_slice(color=True)

# Now we can stitch the dataset properly
stitcher = paprica.tileStitcher(tiles_nissl, overlap_h=20, overlap_v=20)
stitcher.compute_registration()

# We can compare the computed and expected tile positions
paprica.viewer.compare_stitching(stitcher_expected, stitcher, color=True)

# We can then stitch a second channel on the first one:
stitcher_autofluo = paprica.channelStitcher(tiles_nissl, overlap_h=20, overlap_v=20)
stitcher_autofluo.compute_rigid_registration(stitcher)

# Alternativelly you can use the stitching computed for the previous channel on this one (if you have sparse signal
# it might not work to stitch different channels together
stitcher_autofluo = paprica.tileStitcher(tiles_autofluo, overlap_h=20, overlap_v=20)
stitcher_autofluo.database = stitcher.database
stitcher_autofluo.reconstruct_slice(color=True)

# You can also use the viewer to display the stitched tiles
viewer = paprica.tileViewer(tiles_autofluo, stitcher_autofluo.database)
viewer.display_all_tiles()