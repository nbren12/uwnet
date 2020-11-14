from dataclasses import dataclass

import logging
import numpy as np
import os
import xarray as xr

from src.data import assign_apparent_sources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dataset_to_skformat_loader')

@dataclass
class XarrayLoaderSklearn:
    """
    Yield X,y arrays for use in models with sklearn_generic style data formatting
    """
    dataset: xr.Dataset
    batch_size: int
    variables: list = None
    dims: tuple = ('sample', 'time', 'z', 'x', 'y')
    existing_data_file: str = None
    write_sample_to_file: bool = False
    output_data_matrices_path: str = None

    @property
    def batches(self):
        n = self.num_samples
        batch_size = self.batch_size
        indices = list(range(0, n, batch_size)) + [n]
        batches = [slice(i_start, i_end)
                   for i_start, i_end in zip(indices[:-1], indices[1:])]
        return batches

    @property
    def num_samples(self):
        if 'sample' in self.dataset.dims:
            return len(self.dataset.sample)
        else:
            return len(self.dataset.x)

    def __iter__(self):
        """
        :param existing_data_file: if provided, will try to load existing .npy saved data
        :param write_to_file_path: if specified, will write the arrays to writ to
        :return:
        """
        for batch in self.batches:
            logger.info( (f'Load Xarray with n samples = {self.num_samples} ') )

            if self.existing_data_file and os.path.exists(self.existing_data_file):
                logger.info(f'Read existing data from {self.existing_data_file}')
                batch_data = np.load(self.existing_data_file)
                num_target_columns = int((np.shape(batch_data)[1] - 2) / 2)
                x_array = batch_data[:, :num_target_columns]
                y_array = batch_data[:, -num_target_columns:]

            else:
                if self.variables is None:
                    variables = list(self.dataset.data_vars)
                else:
                    variables = self.variables

                subset = self.dataset[variables].isel(x=batch)

                # compute apparent sources Q1 and Q2 for y inputs
                subset = assign_apparent_sources(subset)
                x_array = self._stack_dims_and_concat_feats(
                            subset,
                            ['SLI', 'QT', 'SOLIN', 'SST'])
                y_array = self._stack_dims_and_concat_feats(
                            subset,
                            ['Q1', 'Q2'])
                batch_data = np.hstack([x_array, y_array])

                # rm rows with NaN elements
                batch_data = batch_data[~np.isnan(batch_data).any(axis=1)]
                num_target_columns = int((np.shape(batch_data)[1] - 2) / 2)
                x_array = batch_data[:, :-num_target_columns]
                y_array = batch_data[:, -num_target_columns:]

            if self.write_sample_to_file and self.output_data_matrices_path:
                np.save(self.output_data_matrices_path, batch_data)
                logger.info(f'Wrote batch to {self.output_data_matrices_path}')

            yield x_array, y_array

    def _stack_dims_and_concat_feats(
            self,
            ds,
            variables,
            sample_dims=('x', 'y', 'time'),
            feature_dims=('z')
    ):
        """Convert certain variables of a data frame into 2D numpy arrays"""

        # convert tuple args to lists
        sample_dims = list(sample_dims)
        feature_dims = list(feature_dims)

        flat_arrays = []
        for name in variables:
            da = ds[name]

            # for two-d variables insert a singleton "z" dimension
            if 'z' not in da.dims:
                da = da.expand_dims('z')
            stacked_da = da.stack(samples=sample_dims, features=feature_dims)
            # make sure the rows are samples and columns are features
            transposed_da = stacked_da.transpose('samples', 'features')
            flat_arrays.append(transposed_da.values)

        # concatenate along the final dimension
        return np.concatenate(flat_arrays, axis=1)

