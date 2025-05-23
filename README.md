# Slope


the horn and zeven.npy is the file to do evaluation


Preprcessing:
Both tif file comes from USGC, 10 belongs to 10 meter(as test case), the other one belongs to 1 meter(reference as Ground Truth).

Since 10m vs 1 m, we will use renorm to normalized 10m to be 1 meter version and then do vs.

Comparsion:(In order to compare the result after calculation and Ground Truth)
we will do comparsion in compute_slope which handle the actual input(after preprossing) with the call math differention method horn's and zeven form algorithmethod. This process will create 3 npy files which contain each parts' data after calculation.

Then We will run evaluation.py to evalute these 3 npy files to do the comparison in metrics, visualization chart or any other methods.

Final Image output:
After the experiment comparsion, we can decide to use zeven or horns inside the imageoutput file by calling which one to run the NSW dataset. NSW1m is the original datasets which representing AOI(Area) of bluemountains. the DATA_338389 is the package and extract it will have NSWGovernment - Spatial Services. The orginal data contain 1mm DEM data for the whole region. Since We are works for interaction map, we separate this region to be 4x4 grid which able to do referencing on the target image output.

We used the original dataset(spatial services) into QGIS to help us separate the DEM data to be 16 referencing area, QGIS will reformate the whole dataset to be one vrt which is the dem_aoi.vrt which is the DEM style area map, and then it will based the vrt file to produced the 4x4 grid of area tif files which will be the input for the Imageoutput.py for the final bluemountians slope image output.

The rest of the files is the result either from evaluaiton(comparsion png) or imageoutput(tried different classification methods) which also editable if we want.
