"""
Following is the order of the land cover data:
    Built-up land --> Class 1
    Vegetation --> Class 2
    Water body --> Class 3
    Others --> Class 4
    
"""

####################################################
####################################################
##                                                ##
##              #######       #######             ##
##            ##              ##   ##             ##
##           ##              ##     ##            ##
##           ##             ###########           ##
##           ##            ##         ##          ##
##            ##          ##           ##         ##
##              #######  ####         ####        ##
##                                                ##
####################################################
## Please don't manipulate this part of the code. ##
####################################################

#Importing all necessary libraries
import os, math
import numpy as np
from osgeo import gdal
from copy import deepcopy

#Defining function to read raster file and return array and datasource
def readraster(file):
    dataSource = gdal.Open(file)
    band = dataSource.GetRasterBand(1)
    band = band.ReadAsArray()
    return(dataSource, band)
  
def identicalList(inList):
    global logical
    inList = np.array(inList)
    logical = inList==inList[0]
    if sum(logical) == len(inList):
        return(True)
    else:
        return(False)

def builtupAreaDifference(landcover1, landcover2, buclass=1, cellsize=30):
    return(sum(sum(((landcover2==buclass).astype(int)-(landcover1==buclass).astype(int))!=0))*(cellsize**2)/1000000)
    
#Defining class to read land cover file of two time periods
class landcover():
    
    def __init__(self, file1, file2):
        self.ds_lc1, self.arr_lc1 = readraster(file1)
        self.ds_lc2, self.arr_lc2 = readraster(file2)
        self.performChecks()

    def performChecks(self):
        #check the rows and columns of input land cover datasets
        print("Checking the size of input rasters...")
        if (self.ds_lc1.RasterXSize == self.ds_lc2.RasterXSize) and (self.ds_lc1.RasterYSize == self.ds_lc2.RasterYSize):
            print("Land cover data size matched.")
            self.row, self.col = (self.ds_lc1.RasterYSize, self.ds_lc1.RasterXSize)
        else:
            print("Input land cover files have different height and width.")
        #Check the number of classes in input land cover images
        print("\nChecking feature classes in land cover data...")
        if (self.arr_lc1.max() == self.arr_lc2.max()) and (self.arr_lc1.min() == self.arr_lc2.min()):
            print("The classes in input land cover files are matched.")
            self.nFeature = (self.arr_lc1.max() - self.arr_lc1.min())
        else:
            print("Input land cover data have different class values/ size.")

    def transitionMatrix(self):
        self.tMatrix = np.random.randint(1, size=(self.nFeature, self.nFeature))
        for x in range(0,self.row):
            for y in range(0,self.col):
                t1_pixel = self.arr_lc1[x,y]
                t2_pixel = self.arr_lc2[x,y]
                self.tMatrix[t1_pixel-1, t2_pixel-1] += 1
        self.tMatrixNorm = np.random.randint(1, size=(4,5)).astype(float)
        print("\nTransition Matrix computed, normalisation in progress..")
        #Creating normalised transition matrix
        for x in range(0, self.tMatrix.shape[0]):
            for y in range(0, self.tMatrix.shape[1]):
                self.tMatrixNorm[x,y] = self.tMatrix[x,y]/(self.tMatrix[x,:]).sum()
                
class growthFactors():

    def __init__(self, *args):
        self.gf = dict()
        self.gf_ds = dict()
        self.nFactors = len(args)
        n = 1
        for file in args:
            self.gf_ds[n], self.gf[n] = readraster(file)
            n += 1
        self.performChecks()

    def performChecks(self):
        print("\nChecking the size of input growth factors...")
        rows = []
        cols = []
        for n in range(1, self.nFactors+1):
            rows.append(self.gf_ds[n].RasterYSize)
            cols.append(self.gf_ds[n].RasterXSize)
        if (identicalList(rows) == True) and ((identicalList(cols) == True)):
            print("Input factors have same row and column value.")
            self.row = self.gf_ds[n].RasterYSize
            self.col = self.gf_ds[n].RasterXSize
        else:
            print("Input factors have different row and column value.")

class fitmodel():

    def __init__(self, landcoverClass, growthfactorsClass):
        self.landcovers = landcoverClass
        self.factors = growthfactorsClass
        self.performChecks()
        self.kernelSize = 4

    def performChecks(self):
        print("\nMatching the size of land cover and growth factors...")
        if (self.landcovers.row == self.factors.row) and (self.factors.col == self.factors.col):
            print("Size of rasters matched.")
            self.row = self.factors.row
            self.col = self.factors.col
        else:
            print("ERROR! Raster size not matched please check.")

    def setThreshold(self, builtupThreshold, *OtherThresholdsInSequence):
        self.threshold = list(OtherThresholdsInSequence)
        self.builtupThreshold = builtupThreshold
        if len(self.threshold) == (len(self.factors.gf)):
            print("\nThreshold set for factors")
        else:
            print('ERROR! Please check the number of factors.')

    def predict(self):
        self.predicted = deepcopy(self.landcovers.arr_lc1)
        sideMargin = math.ceil(self.kernelSize/2)
        for y in range(sideMargin,self.row-(sideMargin-1)):
            for x in range(sideMargin,self.col-(sideMargin-1)):
                kernel = self.landcovers.arr_lc1[y-(sideMargin-1):y+(sideMargin), x-(sideMargin-1):x+(sideMargin)]
                builtupCount = sum(sum(kernel==1))
                #If the number of built-up cells greater than equal to assigned threshold
                if (builtupCount >= self.builtupThreshold):
                #if (builtupCount >= self.builtupThreshold) and (self.factors.gf[5][y,x] != 1):  # Adding exception for the restricted areas
                    for factor in range(1,self.factors.nFactors+1):
                #If the assigned thresholds are less than zero, then the smaller than rule applies, else greater than
                        if self.threshold[factor-1] < 0:
                            if (self.factors.gf[factor][y,x] <= abs(self.threshold[factor-1])):
                                self.predicted[y, x] = 1
                            else:
                                pass
                        elif self.threshold[factor-1] > 0:
                            if (self.factors.gf[factor][y,x] >= self.threshold[factor-1]):
                                self.predicted[y, x] = 1
                            else:
                                pass
                if (y%500==0) and (x%500==0):
                    print("Row: %d, Col: %d, Builtup cells count: %d\n" % (y, x, builtupCount), end="\r", flush=True)

    def checkAccuracy(self):
        #Statistical Accuracy
        self.actualBuildup = builtupAreaDifference(self.landcovers.arr_lc1, self.landcovers.arr_lc2)
        self.predictedBuildup = builtupAreaDifference(self.landcovers.arr_lc1, self.predicted)
        self.spatialAccuracy = 100 - (sum(sum(((self.predicted==1).astype(float)-(self.landcovers.arr_lc2==1).astype(float))!=0))/sum(sum(self.landcovers.arr_lc2==1)))*100
        print("Actual growth: %d, Predicted growth: %d" % (self.actualBuildup, self.predictedBuildup))
        #Spatial Accuracy
        print("Spatial accuracy: %f" % (self.spatialAccuracy))

    def exportPredicted(self, outFileName):
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outFileName, self.col, self.row, 1, gdal.GDT_UInt16) # option: GDT_UInt16, GDT_Float32
        outdata.SetGeoTransform(self.landcovers.ds_lc1.GetGeoTransform())
        outdata.SetProjection(self.landcovers.ds_lc1.GetProjection())
        outdata.GetRasterBand(1).WriteArray(self.predicted)
        outdata.GetRasterBand(1).SetNoDataValue(0)        
        outdata.FlushCache() 
        outdata = None

##################################################
## Cellular Automata model ends                 ##
## The below part of the code should be updated ##
##################################################

# Assign the directory where files are located
cwd = os.getcwd()
os.chdir(cwd)

# Input land cover GeoTIFF for two time period
file14 = "LULC_2014.tif"
file15 = "LULC_2015.tif"
file16 = "LULC_2016.tif"
file17 = "LULC_2017.tif"
file18 = "LULC_2018.tif"
file19 = "LULC_2019.tif"
file20 = "LULC_2020.tif"
file21 = "LULC_2021.tif"
file22 = "LULC_2022.tif"

# Input all the parameters
pop14 = "PD_2014.tif"
pop15 = "PD_2015.tif"
pop16 = "PD_2016.tif"
pop17 = "PD_2017.tif"
pop18 = "PD_2018.tif"
pop19 = "PD_2019.tif"
pop20 = "PD_2020.tif"
pop21 = "PD_2021.tif"

wat14 = "Water_2014.tif"
wat15 = "Water_2015.tif"
wat16 = "Water_2016.tif"
wat17 = "Water_2017.tif"
wat18 = "Water_2018.tif"
wat19 = "Water_2019.tif"
wat20 = "Water_2020.tif"
wat21 = "Water_2021.tif"
wat22 = "Water_2022.tif"

slope = "slopeMap.tif"

prox= "primary_proximity.tif"

dem="Blr_DEM_fin.tif"

sim23="simulated2023.tif"
sim24="simulated2024.tif"

sim22_2="simulated2022_2.tif"
sim23_2="simulated2023_2.tif"
sim24_2="simulated2024_2.tif"

# Create a land cover class which takes land cover data for two time period
myLandcover = landcover(sim24_2, file22)

# Create a factors class that configures all the factors for the model
myFactors = growthFactors(pop21,prox,slope,dem)

# Initiate the model with the above created class
caModel = fitmodel(myLandcover, myFactors)

# Set the threshold values
caModel.setThreshold(4,1500,60,-1.8,-900)

# Run the model
caModel.predict()

# Check the accuracy of the predicted values
caModel.checkAccuracy()

# Export the predicted layer
caModel.exportPredicted('simulated2025_2.tif')