from tilemanager import tileMerger
import matplotlib.pyplot as plt

merger = tileMerger(r'/media/sf_shared_folder_virtualbox/multitile_registration/apr/registration_results.csv')
# merger.set_downsample(2)
merger.initialize_merged_array()
merger.merge_additive()

fig, ax = plt.subplots(1, 3)
ax[0].imshow(merger.merged_data[0], cmap='gray')
ax[0].set_title('YX')
ax[1].imshow(merger.merged_data[:, 100, :], cmap='gray')
ax[1].set_title('ZX')
ax[2].imshow(merger.merged_data[:, :, 100], cmap='gray')
ax[0].set_title('ZY')