import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

carToPassengerRatio=(350000/1100000+340000/1050000+370000/1140000+370000/1170000)/4
railtoTrucksRatio=(450000/2570000+480000/2620000+500000/2670000+560000/2730000)/4
cars = numpy.array([785076/0.7*carToPassengerRatio,755398/0.7*carToPassengerRatio,748724/0.7*carToPassengerRatio,641277/0.7*carToPassengerRatio,350000,340000,370000,370000,700000/0.7*carToPassengerRatio,630000/0.7*carToPassengerRatio])
#Interislander makes up for 44% of commercial vehicle freight
trucks = numpy.array([1214681/0.44*(1/(1+railtoTrucksRatio)),1184647/0.44*(1/(1+railtoTrucksRatio)),1209237/0.44*(1/(1+railtoTrucksRatio)),1243969/0.44*(1/(1+railtoTrucksRatio)),2570000,2620000,2670000,2730000,1300000/0.44*(1/(1+railtoTrucksRatio)),1350000/0.44*(1/(1+railtoTrucksRatio))]) #number of truck lane metres = number of commercial lane metres * 1/(1+railToTrucksRatio)
rail = numpy.array([1214681/0.44*(railtoTrucksRatio/(1+railtoTrucksRatio)),1184647/0.44*(railtoTrucksRatio/(1+railtoTrucksRatio)),1209237/0.44*(railtoTrucksRatio/(1+railtoTrucksRatio)),1243969/0.44*(railtoTrucksRatio/(1+railtoTrucksRatio)),450000,480000,500000,560000,1300000/0.44*(railtoTrucksRatio/(1+railtoTrucksRatio)),1350000/0.44*(railtoTrucksRatio/(1+railtoTrucksRatio))]) #number of rail lane metres = number of commercial lane metres * railtoTrucksRatio/(1+railtoTrucksRatio)

xData = numpy.array([2011,2012,2013,2014,2016,2017,2018,2019,2020,2021]) 
yData = cars*2500+trucks*2000+rail*3125

def func(x, a, b, Offset): #logarithmic function with offset
    return  a+b*numpy.log(x) + Offset

# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = func(xData, *parameterTuple)
    return numpy.sum((yData - val) ** 2.0)

def generate_Initial_Parameters():
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)

    parameterBounds = []
    parameterBounds.append([minX, maxX]) # search bounds for a
    parameterBounds.append([minX, maxX]) # search bounds for b
    parameterBounds.append([0.0, maxY]) # search bounds for Offset

    #"seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

#generate initial parameter values
geneticParameters = generate_Initial_Parameters()

#curve fit the test data
fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)

print('Parameters', fittedParameters)

modelPredictions = func(xData, *fittedParameters) 

absError = modelPredictions - yData

SE = numpy.square(absError) #squared errors
MSE = numpy.mean(SE) #mean squared errors
RMSE = numpy.sqrt(MSE) #Root Mean Squared Error, RMSE
Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

#graphics output section
def ModelAndScatterPlot(graphWidth, graphHeight):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    #first the raw data as a scatter plot
    axes.plot(xData, yData,  'D')

    #create data for the fitted equation plot
    xModel = numpy.linspace(min(xData), max(xData))
    yModel = func(xModel, *fittedParameters)

    #now the model as a line plot 
    axes.plot(xModel, yModel)

    axes.set_xlabel('Year') #X axis data label
    axes.set_ylabel('Y Data') #Y axis data label

    plt.show()
    plt.close('all') #clean up after using pyplot

graphWidth = 800
graphHeight = 600
ModelAndScatterPlot(graphWidth, graphHeight)