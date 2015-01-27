# script to compute results

# a new file every 60n + 1 th line

import numpy
import sys

target_file = sys.argv[1] # this will contain both lables and feature (by design)
obtained_lables_file = sys.argv[2] # this will only contain a column of predictions

target_mat = numpy.genfromtxt(target_file,delimiter=',',dtype='float')
ground_truth = target_mat[:,0]
obtained_lables = numpy.genfromtxt(obtained_lables_file,delimiter=',',dtype='float')

num_files = ground_truth.shape[0]/60
all_coefs = numpy.zeros(num_files)

for i in range(0, num_files):
	cur_ground_truth = ground_truth[60*i+1:60*(i+1)]
	cur_obt_lables = obtained_lables[60*i+1:60*(i+1)]
	X = numpy.hstack((cur_ground_truth,cur_obt_lables))
	corr_coef = numpy.corrcoef(X)[0,1]
	print corr_coef
	all_coefs[i] = corr_coef

mean_result = numpy.mean(all_coefs)



