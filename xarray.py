import numpy as np
import xarray as xr
cityOrig = ['chicago','new york', 'boston']
cityDest = ['chicago','new york', 'boston']
month = ['jan','feb','mar','apr']
data = np.random.rand(4,3,3,4)

myArray = xr.DataArray(data,
                       dims=['dat','orig','dest','month'],
                       coords = {'orig':cityOrig,'dest':cityDest,'month':month})

print(myArray[:,1,2,1].data)
print(myArray.loc[:,'chicago','new york','jan'].data)



                       coords[,cityOrig,cityDest,month], dims = ['','orig','dest','month'])

state =


In [2]: locs = ['IA', 'IL', 'IN']

In [3]: times = pd.date_range('2000-01-01', periods=4)

In [4]: foo = xr.DataArray(data, coords=[times, locs], dims=['time', 'space'])

In [5]: foo
Out[5]:
<xarray.DataArray (time: 4, space: 3)>
array([[0.12697 , 0.966718, 0.260476],
       [0.897237, 0.37675 , 0.336222],
       [0.451376, 0.840255, 0.123102],
       [0.543026, 0.373012, 0.447997]])
Coordinates:
  * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03 2000-01-04
  * space    (space) <U2 'IA' 'IL' 'IN'

In [1]: da = xr.DataArray(np.random.rand(4, 3),
   ...:                   [('time', pd.date_range('2000-01-01', periods=4)),
   ...:                    ('space', ['IA', 'IL', 'IN'])])
   ...:

In [2]: da[:2]
Out[2]:
<xarray.DataArray (time: 2, space: 3)>
array([[0.12697 , 0.966718, 0.260476],
       [0.897237, 0.37675 , 0.336222]])
Coordinates:
  * time     (time) datetime64[ns] 2000-01-01 2000-01-02
  * space    (space) <U2 'IA' 'IL' 'IN'

In [3]: da[0, 0]
Out[3]:
<xarray.DataArray ()>
array(0.12697)
Coordinates:
    time     datetime64[ns] 2000-01-01
    space    <U2 'IA'

In [4]: da[:, [2, 1]]
Out[4]:
<xarray.DataArray (time: 4, space: 2)>
array([[0.260476, 0.966718],
       [0.336222, 0.37675 ],
       [0.123102, 0.840255],
       [0.447997, 0.373012]])
Coordinates:
  * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03 2000-01-04
  * space    (space) <U2 'IN' 'IL'