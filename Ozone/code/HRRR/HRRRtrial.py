from herbie import Herbie
H = Herbie(
    "2021-07-01 12:00",  # model run date
    model="hrrr",  # model name
    product="sfc",  # model produce name (model dependent)
    fxx=6,  # forecast lead time
)

H.inventory(searchString="TMP")

ds = H.xarray(":(?:TMP|RH):2 m", remove_grib=False)
ds
ds.t2m.plot()