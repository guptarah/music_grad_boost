# script to train weak classifiers

import numpy
import sys

def compute_error (features,targets,feat_inds,all_coeffs,classf_id):
	targets = numpy.matrix(targets).T
	computed_output = numpy.zeros((targets.shape[0],1))

	for k in range(0,classf_id+1):
		cur_coeffs = all_coeffs[k,:]
		cur_feat = feat_inds[k]
		cur_feat_vals = features[:,cur_feat[0]]
		# make matrix for computing outputs
	
		num_files = targets.shape[0]/60
		feat_win_len = all_coeffs.shape[1]
		num_feat_wins = num_files * 60
		feat_mat = numpy.zeros((num_feat_wins,feat_win_len))
			
		for i in range(0,num_files):
	                cur_file_feat_vals = cur_feat_vals[60*i:60*(i+1)]
        	        # making matrix
                	for j in range(0,60):
                        	if j < filt_len:
                                	cur_file_feat_vals_ext = numpy.hstack((cur_file_feat_vals[0]*numpy.ones(filt_len),cur_file_feat_vals))
	                                cur_feat_window = cur_file_feat_vals_ext[j:j+2*filt_len+1]
        	                elif j+filt_len >= 60:
                	                cur_file_feat_vals_ext = numpy.hstack((cur_file_feat_vals,cur_file_feat_vals[-1]*numpy.ones(filt_len)))
                        	        cur_feat_window = cur_file_feat_vals_ext[j-filt_len:j+filt_len+1]
	                        else:
        	                        cur_feat_window = cur_file_feat_vals[j-filt_len:j+filt_len+1]
                	        
				feat_mat[60*i + j,:] = cur_feat_window

		cur_outputs = feat_mat * numpy.matrix(cur_coeffs).T
		# compute the output
		computed_output = computed_output + cur_outputs 
		
	error = numpy.sqrt((computed_output - targets).T * (computed_output - targets)/ targets.shape[0])
	return error

def compute_gamma(output,residual):
	# right now only doing grid search (ans expected to be close to 1)
	r = 0
	error = numpy.zeros(20)
	while r < 20:
		output_temp = float(r)/10 * output
		error[r] = numpy.sqrt((output_temp - residual).T * (output_temp - residual)/output_temp.shape[0])
		r = r + 1
	chosen_r = numpy.argmin(error)
	gamma = float(chosen_r)/10
	return gamma


def compute_filter_given_feat(residual,train_features,train_targets,cur_feat_ind,filt_len):
        cur_feat_vals = train_features[:,cur_feat_ind]

        num_train_files = train_targets.shape[0]/60 # starting to make feature matrix
        feat_win_len = 2*filt_len + 1
        num_feat_wins = num_train_files * 60 
        feat_mat = numpy.zeros((num_feat_wins,feat_win_len))
        target = numpy.zeros((num_feat_wins,1))

        for i in range(0,num_train_files):
                cur_file_feat_vals = cur_feat_vals[60*i:60*(i+1)]
                cur_file_residual = residual[60*i:60*(i+1)]
                # making matrix
                for j in range(0,60): 
                        target[i*60 + j] = cur_file_residual[j]
			if j < filt_len:
				cur_file_feat_vals_ext = numpy.hstack((cur_file_feat_vals[0]*numpy.ones(filt_len),cur_file_feat_vals))
				cur_feat_window = cur_file_feat_vals_ext[j:j+2*filt_len+1]	
			elif j+filt_len >= 60:
				cur_file_feat_vals_ext = numpy.hstack((cur_file_feat_vals,cur_file_feat_vals[-1]*numpy.ones(filt_len)))
				cur_feat_window = cur_file_feat_vals_ext[j-filt_len:j+filt_len+1]
			else:
	                        cur_feat_window = cur_file_feat_vals[j-filt_len:j+filt_len+1]
       
	                feat_mat[60*i + j,:] = cur_feat_window

        # get MMSE based coefficients
        w = numpy.linalg.lstsq(feat_mat,target)
        coeffs = numpy.matrix(w[0].T)

        # get outputs based on these coeffs
        classf_output = feat_mat * coeffs.T

        return coeffs,cur_feat_ind,classf_output


train_file = sys.argv[1] # train file 
dev_file = sys.argv[2] # dev file
test_file = sys.argv[3]
num_classifiers = int(sys.argv[4])

train_data = numpy.genfromtxt(train_file,delimiter=',') 
dev_data = numpy.genfromtxt(dev_file,delimiter=',')
test_data = numpy.genfromtxt(test_file,delimiter=',')

train_targets = train_data[:,0]
train_features = train_data[:,1:]
num_train_files = train_targets.shape[0]/60
num_feats = train_features.shape[1]

dev_targets = dev_data[:,0]
dev_features = dev_data[:,1:] 
num_dev_files = dev_targets.shape[0]/60

test_targets = test_data[:,0]
test_features = test_data[:,1:]
num_test_files = test_targets.shape[0]/60

filt_lens = [2,5,10,20] # this len will be doubled as this is one sided length
filt_len = 2

feat_inds = numpy.zeros((num_classifiers,1))
gammas = numpy.zeros((num_classifiers,1))
all_coeffs = numpy.zeros((num_classifiers,2*filt_len+1))

train_energy = numpy.sqrt(numpy.matrix(train_targets) * numpy.matrix(train_targets).T / train_targets.shape[0])
dev_energy = numpy.sqrt(numpy.matrix(dev_targets) * numpy.matrix(dev_targets).T / dev_targets.shape[0])
test_energy = numpy.sqrt(numpy.matrix(test_targets) * numpy.matrix(test_targets).T / test_targets.shape[0])
print 'signal energies ',train_energy, dev_energy, test_energy 

for classf_id in range(0,num_classifiers):
	
	print ''
	print 'iter number: ', classf_id

	if classf_id == 0:	
		# when no weak classifier exists (residual = targets)
		residual = numpy.matrix(train_targets).T

	print 'scanning each feature for performance'
	temp_all_coeffs = all_coeffs
	temp_feat_inds = feat_inds
	dev_errors = numpy.zeros(num_feats)
	for cur_feat_ind in range(0,num_feats):	
		[cur_coeffs,feat_ind,classf_output] = compute_filter_given_feat(residual,train_features,train_targets,cur_feat_ind,filt_len)
		remaining_error = classf_output - residual 
		print 'train error', numpy.sqrt(numpy.matrix(remaining_error).T * numpy.matrix(remaining_error) / remaining_error.shape[0])
		temp_all_coeffs[classf_id,:] = cur_coeffs
		temp_feat_inds[classf_id,:] = feat_ind
		# compute results on the dev set
		error_dev = compute_error(dev_features,dev_targets,temp_feat_inds,temp_all_coeffs,classf_id)
		print 'feature in question: ', cur_feat_ind, ' error: ',error_dev
		dev_errors[cur_feat_ind] = error_dev
	chosen_feat_ind = numpy.argmin(dev_errors)
	print 'chosen feat: ', chosen_feat_ind

	[cur_coeffs,feat_ind,classf_output] = compute_filter_given_feat(residual,train_features,train_targets,chosen_feat_ind,filt_len)
	all_coeffs[classf_id,:] = cur_coeffs
	feat_inds[classf_id,0] = chosen_feat_ind 

#	print 'computing gamma'
#	# compute gamma
#	if classf_id == 0:
#		gamma = 1
#	else:
#		gamma = compute_gamma(classf_output,residual)

	residual = residual - 1*classf_output # gamma always turns out to be 1

	# compute the results on the dev set
	error_dev = compute_error (dev_features,dev_targets,feat_inds,all_coeffs,classf_id) 
	print 'dev error ',error_dev

	# compute the results on the test set
	error_test = compute_error(test_features,test_targets,feat_inds,all_coeffs,classf_id)
	print 'test error',error_test	
