# script to get results on a supplied file

import numpy
import sys


def filter_predictions(predictions,filt_len):

	if numpy.mod(filt_len,2) == 0:
		print 'filter length even, extending by 1'
		filt_len = filt_len + 1	

	num_files = numpy.matrix(predictions).shape[0]/60
	filt_preds = numpy.zeros(predictions.shape)
	for i in range(0,num_files):
		cur_preds = predictions[i*60:(i+1)*60]
		to_concat_begin = numpy.ones((filt_len,1))*cur_preds[0]
		to_concat_end = numpy.ones((filt_len,1))*cur_preds[-1]
		filter_input = numpy.concatenate((to_concat_begin,cur_preds,to_concat_end),axis=0)	
		filtered_output = numpy.convolve(numpy.array(filter_input.T)[0],numpy.ones((filt_len,))/filt_len)
		cut_length = filt_len + filt_len/2
		filt_preds[i*60:(i+1)*60,0] = filtered_output[cut_length:-1*cut_length]	
	
	return filt_preds


input_file = sys.argv[1]
coeff_file = sys.argv[2]
filt_len = int(sys.argv[3])
#coeff_file_pinv = sys.argv[3]
#coeff_file_matlab = sys.argv[4]

input_data = numpy.genfromtxt(input_file,delimiter=',',dtype='float')
input_coeff = numpy.genfromtxt(coeff_file,delimiter=',',dtype='float')
#input_coeff_pinv = numpy.genfromtxt(coeff_file_pinv,delimiter=',',dtype='float')
#input_coeff_matlab = numpy.genfromtxt(coeff_file_matlab,delimiter=',',dtype='float')

targets = numpy.matrix(input_data[:,0]).T
input_features = input_data[:,1:]
predictions = input_features * numpy.matrix(input_coeff).T
predictions[numpy.abs(predictions) > 1 ] = 0 # Process predictions to remove outliers
predictions_filt = filter_predictions(predictions,filt_len)  
#predictions_pinv = input_features * numpy.matrix(input_coeff_pinv).T
#predictions_matlab = input_features * numpy.matrix(input_coeff_matlab).T
to_save = numpy.concatenate((targets,predictions,predictions_filt),axis=1)

#num_files = targets.shape[0]/60
#all_coefs = numpy.zeros(num_files)

#for i in range(0, num_files):
#	cur_ground_truth = targets[60*i+1:60*(i+1)]
#        cur_obt_lables = predictions[60*i+1:60*(i+1)]
#        X = numpy.hstack((cur_ground_truth,cur_obt_lables))
#        corr_coef = numpy.corrcoef(X.T)[0,1]
#        print corr_coef
#        all_coefs[i] = corr_coef
#
#mean_result = numpy.mean(all_coefs)

# compute the RMSE
error = targets - predictions_filt
print error.shape
print to_save.shape
#error_pinv = targets - predictions_pinv
#error_matlab = targets - predictions_matlab
#to_save = numpy.concatenate((to_save,error,error_pinv,error_matlab),axis=1)
to_save = numpy.concatenate((to_save,error),axis=1)
numpy.savetxt('temp_pred',to_save,fmt='%f')

print error.shape[0]
rmse = numpy.sqrt(error.T * error / error.shape[0])
#rmse_pinv = numpy.sqrt(error_pinv.T * error_pinv / error_pinv.shape[0])
#rmse_matlab = numpy.sqrt(error_matlab.T * error_matlab / error_matlab.shape[0])
energy_targets = numpy.sqrt(targets.T * targets / targets.shape[0])
#print 'rmse: ',rmse,rmse_pinv,rmse_matlab,' energy of targets: ', energy_targets
print 'rmse: ',rmse,' energy of targets: ', energy_targets
