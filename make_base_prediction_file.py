import numpy
import sys

lable_file = sys.argv[1]
feature_file = sys.argv[2]

lables = numpy.genfromtxt(lable_file,delimiter=',',dtype='float')
features = numpy.genfromtxt(feature_file,delimiter=';',dtype='float')

if features.shape[0] == 60:
	lables = lables[0:60]
else:
	features = features[0:60,:]
	lables = lables[0:60]


mean_feats = numpy.mean(features, axis=0)
tiled_mean_feats = numpy.tile(mean_feats,(60,1))

print tiled_mean_feats.shape 
print numpy.matrix(lables).T.shape
print features.shape

data_mat_base = numpy.hstack((numpy.matrix(lables).T,tiled_mean_feats))
data_mat_framewise = numpy.hstack((numpy.matrix(lables).T,features)) 
numpy.savetxt('cur_data',data_mat_base,fmt='%5f',delimiter=',')
numpy.savetxt('cur_data_fw',data_mat_framewise,fmt='%5f',delimiter=',')


