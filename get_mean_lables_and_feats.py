# script to get mean of lable and features per file for lr

import sys
import numpy

input_file = sys.argv[1]
output_file = sys.argv[2]

input_data = numpy.genfromtxt(input_file,delimiter=',')
targets = numpy.matrix(input_data[:,0]).T
features = input_data[:,1:]
num_files = input_data.shape[0]/60
output_mat = numpy.zeros((num_files,features.shape[1]+1))

for i in range(0,num_files):
	cur_ground_truth = targets[60*i:60*(i+1)]
	cur_feats = features[60*i:60*(i+1),:]
	mean_ground_truth = numpy.mean(cur_ground_truth)
	mean_feats = numpy.mean(cur_feats,axis=0)
	output_mat[i,0] = mean_ground_truth
	output_mat[i,1:] = mean_feats

numpy.savetxt(output_file,output_mat,fmt='%.18e', delimiter=',')
		
