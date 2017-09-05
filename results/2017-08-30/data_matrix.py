class DataMatrix(object):
    """Matrix for inputting/outputting datamatrices from xarray dataset objects"""
    
    def __init__(self, feature_dims, sample_dims, variables):
        self.feature_dims = feature_dims
        self.sample_dims = sample_dims
        self.variables = variables


    def dataset_to_mat(self, X):
        Xs = X.stack(samples=self.sample_dims, features=self.feature_dims).transpose('samples', 'features')
        
        self._var_coords = {k: Xs[k].coords for k in self.variables}
        
        # store offsets
        offset = 0
        self._var_slices = {}
        one_sample = Xs.isel(samples=0)
        for k in self.variables:
            nfeat = one_sample[k].size
            self._var_slices[k] = slice(offset, offset+nfeat)
            offset += nfeat
            
        return np.hstack(self.column_var(Xs[k]) for k in self.variables)
    
    def mat_to_dataset(self, X):
        
        data_dict = {}
        
        for k in self.variables:
            sl = self._var_slices[k]
            data = X[:,self._var_slices[k]]
            
            if data.shape[1] == 1:
                data = data[:,0]
        
            data_dict[k] = xr.DataArray(data, self._var_coords[k], name=k)
        return xr.Dataset(data_dict)

    def column_var(self, x):
        if x.ndim == 1:
            return x.data[:,None]
        else:
            return x

        
        
def test_datamatrix():
    mat = DataMatrix(['z'], ['time'], ['qt', 'sl', 'LHF', 'Prec'])
    y = mat.dataset_to_mat(inputs)
    
    x = mat.mat_to_dataset(y)
    
    
    for k in inputs.data_vars:
        np.testing.assert_allclose(inputs[k], x[k])
