from PIL import Image
import numpy as np
traffic = Image.open(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\trafficData\averageTrafficDay.tif")
traffic = np.array(traffic)

traffic.shape

