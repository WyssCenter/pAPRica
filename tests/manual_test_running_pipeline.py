import pipapr
import pyapr
import napari


rp = pipapr.clearscopeRunningPipeline(path='./data/synthetic/tif',
                                      n_channels=1)
rp.activate_conversion()
rp.run()

