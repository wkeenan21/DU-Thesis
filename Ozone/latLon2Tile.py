import math
def coord2tile(lat, long, zoom):
    n = 2 ^ zoom
    xtile = n * ((long + 180) / 360)
    ytile = n * (1 - (math.log(math.tan(lat) + math.acos(lat)) / math.pi)) / 2
    return [xtile, ytile]



import math
def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  return (xtile, ytile)

tile = deg2num(39.73, -104.99, 12)
print(tile)