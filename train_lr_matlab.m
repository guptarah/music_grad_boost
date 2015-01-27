function train_lr_matlab(data_file)

data = load(data_file);
data_lables = data(:,1);
data_features = data(:,2:end);

w = pinv(data_features)*data_lables;
csvwrite('lr_coeffs_matlab',w);


% loading the python coeffs
lr_coeffs_python = load('lr_coeffs_python');
lr_coeffs_python_pinv = load('lr_coeffs_python_pinv');

error_matlab = data_lables - data_features * w;
error_python = data_lables - data_features * lr_coeffs_python;
error_python_pinv = data_lables - data_features * lr_coeffs_python_pinv;

sqrt(error_matlab' * error_matlab / length(error_matlab))
sqrt(error_python' * error_python / length(error_python))
sqrt(error_python_pinv' * error_python_pinv / length(error_python_pinv))
