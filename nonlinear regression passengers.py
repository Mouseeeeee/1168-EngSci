import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

xData = numpy.array([2011,2012,2013,2014,2016,2017,2018,2019,2020,2021])
yData = numpy.array([785076/0.7,755398/0.7,748724/0.7,641277/0.7,1100000,1050000,1140000,1170000,700000/0.7,630000/0.7]) #Interislander makes up for 70% of passengers

def func(x, a, b, Offset): #logarithmic function with offset
    return  a+b*numpy.log(x) + Offset

#function for genetic algorithm to minimise sum of squared error
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") #do not print warnings by genetic algorithm
    val = func(xData, *parameterTuple)
    return numpy.sum((yData - val) ** 2.0)

def generate_Initial_Parameters():
    #min and max used for bounds
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

#graphics output
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
    axes.set_ylabel('Number of Passengers') #Y axis data label

    plt.show()
    plt.close('all') #clean up after using pyplot

graphWidth = 800
graphHeight = 600
ModelAndScatterPlot(graphWidth, graphHeight)