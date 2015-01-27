import numpy
import sys

data_file = sys.argv[1]
matlab_lr_coeffs = sys.argv[2]

data = numpy.genfromtxt(data_file,delimiter=',',dtype='float')

data_lables = numpy.matrix(data[:,0]).T
data_features = data[:,1:]

print data_features.shape,data_lables.shape

w = numpy.linalg.lstsq(data_features,data_lables)
coeffs = w[0]

coeffs_pinv = numpy.linalg.pinv(data_features)*data_lables
coeffs_matlab = numpy.matrix(numpy.genfromtxt(matlab_lr_coeffs)).T

error_train = data_lables - data_features*coeffs
error_train_pinv = data_lables - data_features*coeffs_pinv
error_matlab = data_lables - data_features*coeffs_matlab

error_train = numpy.sqrt(error_train.T * error_train/error_train.shape[0])
error_train_pinv = numpy.sqrt(error_train_pinv.T * error_train_pinv/error_train_pinv.shape[0])
error_train_matlab = numpy.sqrt(error_matlab.T * error_matlab/error_matlab.shape[0])


print error_train
print error_train_pinv
print error_train_matlab

numpy.savetxt('lr_coeffs_python', coeffs, fmt='%.18e', delimiter=' ')
numpy.savetxt('lr_coeffs_python_pinv', coeffs_pinv, fmt='%.18e',delimiter=' ')
 
