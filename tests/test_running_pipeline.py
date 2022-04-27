import pipapr
import pyapr
import napari


# rp = pipapr.clearscopeRunningPipeline(path='/media/hbm/SSD1/ClearScope/test/220108-08-TJ',
#                                       n_channels=1)
# rp.activate_conversion()
# rp.run()

path = '/media/hbm/SSD1/ClearScope/test/220108-08-TJ/0001/APR/ch0/0_0.apr'

toto = pyapr.data_containers.LazySlicer(path)

v = napari.Viewer()
# napari.run()

v.add_image(toto)
napari.run()
import time
time.sleep(20)