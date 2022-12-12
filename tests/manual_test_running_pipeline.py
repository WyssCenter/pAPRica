import paprica
import pyapr
import napari


def main():
    rp = paprica.clearscopeRunningPipeline(path='./data/synthetic/tif',
                                           n_channels=1)
    rp.activate_conversion()
    rp.run()


if __name__ == '__main__':
  main()