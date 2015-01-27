import numpy
import sys

data_file = sys.argv[1]
data = numpy.genfromtxt(data_file,delimiter=',',dtype='float')

data_lables = data[:,0]
data_features = data[:,1:]

print data_features.shape,data_lables.shape

w = numpy.linalg.lstsq(data_features,data_lables)
coeffs = w[0]

coeffs_pinv = numpy.linalg.pinv(data_features)*data_lables

numpy.savetxt('base_coeffs_python', coeffs, fmt='%.18e', delimiter=' ')
numpy.savetxt('base_coeffs_python_pinv', coeffs_pinv, fmt='%.18e', delimiter=' ')

 
