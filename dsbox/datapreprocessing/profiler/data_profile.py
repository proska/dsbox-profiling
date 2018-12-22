import json
import sys

import time
import typing
from collections import defaultdict

import d3m.metadata.base as mbase
import numpy as np
import pandas as pd
import pytypes
import logging

from dsbox.datapreprocessing.profiler.date_featurizer_org import DateFeaturizerOrg
from dsbox.datapreprocessing.profiler.spliter import PhoneParser, PunctuationParser, NumAlphaParser

from . import category_detection
from . import dtype_detector
from . import feature_compute_hih as fc_hih
from . import feature_compute_lfh as fc_lfh

_logger = logging.getLogger(__name__)

MetaData_T = typing.Dict[str, typing.Any]

VERBOSE = 1

computable_metafeatures = [
    'ratio_of_values_containing_numeric_char', 'ratio_of_numeric_values',
    'number_of_outlier_numeric_values', 'num_filename', 'number_of_tokens_containing_numeric_char',
    'number_of_numeric_values_equal_-1', 'most_common_numeric_tokens', 'most_common_tokens',
    'ratio_of_distinct_tokens', 'number_of_missing_values',
    'number_of_distinct_tokens_split_by_punctuation', 'number_of_distinct_tokens',
    'ratio_of_missing_values', 'semantic_types', 'number_of_numeric_values_equal_0',
    'number_of_positive_numeric_values', 'most_common_alphanumeric_tokens',
    'numeric_char_density', 'ratio_of_distinct_values', 'number_of_negative_numeric_values',
    'target_values', 'ratio_of_tokens_split_by_punctuation_containing_numeric_char',
    'ratio_of_values_with_leading_spaces', 'number_of_values_with_trailing_spaces',
    'ratio_of_values_with_trailing_spaces', 'number_of_numeric_values_equal_1',
    'natural_language_of_feature', 'most_common_punctuations', 'spearman_correlation_of_features',
    'number_of_values_with_leading_spaces', 'ratio_of_tokens_containing_numeric_char',
    'number_of_tokens_split_by_punctuation_containing_numeric_char', 'number_of_numeric_values',
    'ratio_of_distinct_tokens_split_by_punctuation', 'number_of_values_containing_numeric_char',
    'most_common_tokens_split_by_punctuation', 'number_of_distinct_values',
    'pearson_correlation_of_features',
    'semantic_types']

default_metafeatures = [
    'ratio_of_values_containing_numeric_char', 'ratio_of_numeric_values',
    'number_of_outlier_numeric_values', 'num_filename', 'number_of_tokens_containing_numeric_char',
    'semantic_types']


class Profiler:
    """
    data profiler moduel. Now only supports csv data.

    Parameters:
    ----------
    _punctuation_outlier_weight: a integer
        the coefficient used in outlier detection for punctuation. default is 3

    _numerical_outlier_weight

    _token_delimiter: a string
        delimiter that used to separate tokens, default is blank space " ".

    _detect_language: boolean
        true: do detect language; false: not detect language

    _topk: a integer

    _verbose: boolean
        control the _verbose

    Attributes:
    ----------
    """

    def __init__(self, metafeatures=default_metafeatures) -> None:
        # super().__init__(hyperparams=hyperparams)

        # All other attributes must be private with leading underscore
        self._punctuation_outlier_weight = 3
        self._numerical_outlier_weight = 3
        self._token_delimiter = " "
        self._detect_language = False
        self._topk = 10
        self._verbose = VERBOSE
        self._DateFeaturizer = None
        # list of specified features to compute
        self._specified_features = metafeatures

        self._PhoneParser = None

    @staticmethod
    def flatten_dict(d, prefix: str = ""):
        out = {}
        for k, v in d.items():
            # print(k,"___:____", v)
            if isinstance(v, set):
                if len(v) == 0:
                    continue
                if len(v) > 1:
                    print("MULTI SET")
                out[prefix + k] = list(v)[0]
            elif isinstance(v, dict):
                d_c = Profiler.flatten_dict(v, prefix=k + "_")
                for k_c, v_c in d_c.items():
                    out[k_c] = v_c
            elif isinstance(v, list):
                for d_c in v:
                    if isinstance(d_c, dict) and 'name' in d_c:
                        d_c_copy = d_c.copy()
                        pr = d_c_copy.pop('name')
                        d_c_t = Profiler.flatten_dict(d_c_copy, prefix=k + f"_{pr}_")
                        for k_c_c, v_c_c in d_c_t.items():
                            out[k_c_c] = v_c_c
            else:
                out[prefix + k] = v
        return out

    def vectorize_metadata(self, metadata: MetaData_T) -> typing.Dict[str, pd.Series]:
        col_names = list(metadata.keys())
        # res = pd.DataFrame(None, index=col_names)
        f_c = {}
        for col in col_names:
            f_c[col] = pd.Series(Profiler.flatten_dict(metadata[col]))

        return f_c


    def produce(self, inputs: pd.DataFrame) -> typing.Dict[str, typing.Any]:
        """
        generate features for the input.
        Input:
            typing.Union[container.Dataset, container.DataFrame, container.ndarray,
            container.matrix, container.List]
        Output:
            typing.Union[container.Dataset, container.DataFrame, container.ndarray,
            container.matrix, container.List]
        """
        metadata: MetaData_T = {}

        # calling the utility to detect integer and float datatype columns
        metadata = dtype_detector.detect_numbers(inputs, metadata)

        # calling date detect_numbers
        date_meta = self._detect_dates(inputs, metadata)

        # calling the utility to categorical datatype columns
        metadata = self._produce(inputs, metadata)
        # I guess there are updating the metdata here
        # inputs.metadata = metadata

        # calling the PhoneParser detect_numbers
        _sample_df = self.sampling_df(inputs)
        self._PhoneParser = PhoneParser(_sample_df)

        assert len(_sample_df) > 0
        phone_parser_indices = self._PhoneParser.detect()
        if phone_parser_indices:
            for i in phone_parser_indices:
                col_name = inputs.columns.values[i]
                # old_metadata = metadata[col_name]
                # print("old metadata", old_metadata)
                if 'https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber' not in \
                        metadata[col_name]["semantic_types"]:
                    metadata[col_name]["semantic_types"] += ('AmericanPhoneNumber',)
                if 'https://metadata.datadrivendiscovery.org/types/UnnormalizedEntity' not in \
                        metadata[col_name]["semantic_types"]:
                    metadata[col_name]["semantic_types"] += ('UnnormalizedEntity',)

                if isinstance(_sample_df.iloc[:, i].head(1).values[0], str):
                    metadata[col_name]["structural_type"] = type("str")
                elif isinstance(_sample_df.iloc[:, i].head(1).values[0], int):
                    metadata[col_name]["structural_type"] = type(10)
                else:
                    metadata[col_name]["structural_type"] = type(10.2)

                # inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, i), old_metadata)
                # metadata[col_name] = dict(**metadata[col_name], )

        # calling the PunctuationSplitter detect_numbers
        self._PunctuationSplitter = PunctuationParser(_sample_df)

        PunctuationSplitter_indices = self._PunctuationSplitter.detect()
        if PunctuationSplitter_indices[0]:
            for i in PunctuationSplitter_indices[0]:
                old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i)))
                if 'https://metadata.datadrivendiscovery.org/types/CanBeSplitByPunctuation' not in \
                        old_metadata["semantic_types"]:
                    old_metadata["semantic_types"] += (
                        'https://metadata.datadrivendiscovery.org/types/CanBeSplitByPunctuation',)

                if isinstance(_sample_df.iloc[:, i].head(1).values[0], str):
                    old_metadata["structural_type"] = type("str")
                elif isinstance(_sample_df.iloc[:, i].head(1).values[0], int):
                    old_metadata["structural_type"] = type(10)
                else:
                    old_metadata["structural_type"] = type(10.2)

                # _logger.info(
                #     "Punctuation detect_numbers. 'column_index': '%(column_index)d', "
                #     "'old_metadata': '%(old_metadata)s', 'new_metadata': '%(new_metadata)s'",
                #     {
                #         'column_index': i,
                #         'old_metadata': dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i))),
                #         'new_metadata': old_metadata,
                #     },
                # )
                inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, i), old_metadata)

        # calling the NumAlphaSplitter detect_numbers

        self._NumAlphaSplitter = NumAlphaParser(_sample_df)

        NumAlphaSplitter_indices = self._NumAlphaSplitter.detect()

        if NumAlphaSplitter_indices[0]:
            for i in NumAlphaSplitter_indices[0]:
                old_metadata = dict(inputs.metadata.query((mbase.ALL_ELEMENTS, i)))
                if 'https://metadata.datadrivendiscovery.org/types/CanBeSplitByAlphanumeric' not \
                        in \
                        old_metadata["semantic_types"]:
                    old_metadata["semantic_types"] += (
                        'https://metadata.datadrivendiscovery.org/types/CanBeSplitByAlphanumeric',)

                if isinstance(_sample_df.iloc[:, i].head(1).values[0], str):
                    old_metadata["structural_type"] = type("str")
                elif isinstance(_sample_df.iloc[:, i].head(1).values[0], int):
                    old_metadata["structural_type"] = type(10)
                else:
                    old_metadata["structural_type"] = type(10.2)

                inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, i), old_metadata)

        return self.vectorize_metadata(metadata)

    def sampling_df(self, inputs):
        # cleaned_df = inputs.dropna()
        _sample_df = inputs.iloc[0:min(inputs.shape[0], 50), :]
        assert len(_sample_df) > 0, f"{_sample_df.shape}"
        return _sample_df

    def _detect_dates(self, inputs: pd.DataFrame, metadata: MetaData_T):
        self._DateFeaturizer = DateFeaturizerOrg(inputs)
        _sample_df = inputs.dropna().iloc[0:min(inputs.shape[0], 50), :]
        cols = self._DateFeaturizer.detect_date_columns(_sample_df)
        if cols:
            indices = [inputs.columns.get_loc(c) for c in cols if c in inputs.columns]
            for i in indices:
                col_metadata = metadata[i]
                temp_value = list(col_metadata["semantic_types"])
                if len(temp_value) >= 1:
                    if 'categorical' not in col_metadata["semantic_types"]:
                        col_metadata["semantic_types"] += ('categorical',)
                    if 'time' not in col_metadata["semantic_types"]:
                        col_metadata["semantic_types"] += ('time',)
                if isinstance(_sample_df.iloc[:, i].head(1).values[0], str):
                    col_metadata["structural_type"] = type("str")
                elif isinstance(_sample_df.iloc[:, i].head(1).values[0], int):
                    col_metadata["structural_type"] = type(10)
                else:
                    col_metadata["structural_type"] = type(10.2)

                metadata[i] = col_metadata

    def _produce(self, inputs: pd.DataFrame, metadata: MetaData_T) -> MetaData_T:
        """
        Parameters:
        -----------
        Input:
            typing.Union[container.Dataset, container.DataFrame, container.ndarray,
            container.matrix, container.List]
        metadata: DataMetadata
            Store generate metadata. If metadata is None, then inputs must be container,
            which has a metadata field to store the generated data.
        prefix: Selector
            Selector prefix into metadata

        """
        # if isinstance(inputs, container.Dataset):
        #     for table_id, resource in inputs.items():
        #         prefix = prefix + [table_id]
        #         metadata = self._produce(resource, metadata, prefix)
        # elif isinstance(inputs, list):
        #     for index, item in enumerate(inputs):
        #         metadata = self._produce(item, metadata, prefix + [index])
        # elif isinstance(inputs, pd.DataFrame):
        #     metadata = self._profile_data(inputs, metadata, prefix)
        # elif isinstance(inputs, np.matrix) or (
        #         isinstance(inputs, np.ndarray) and len(inputs.shape) == 2):
        #     df = pd.DataFrame(inputs)
        #     metadata = self._profile_data(df, metadata, prefix)
        # elif isinstance(inputs, container.ndarray):
        #     metadata = self._profile_ndarray(inputs, metadata, prefix)

        metadata = self._profile_data(inputs, metadata)

        return metadata

    def _profile_ndarray(self, array, metadata, prefix):
        # TODO: What to do with ndarrays?
        return metadata

    def _profile_data(self, data_in, metadata):

        """
        Main function to profile the data. This functions will
        1. calculate features
        2. update metadata with features

        Parameters
        ----------
        data: pandas.DataFrame that needs to be profiled
        ----------
        """
        data = data_in.copy(deep=True)
        if self._verbose:
            print("====================have a look on the data: ====================\n")
            print(data.head(2))

        # calculations
        if self._verbose:
            print("====================calculating the features ... ====================\n")

        # STEP 1: data-level calculations
        if "pearson_correlation_of_features" in self._specified_features:
            corr_pearson = data.corr()
            corr_columns = list(corr_pearson.columns)
            corr_id = [data.columns.get_loc(n) for n in corr_columns]

        if "spearman_correlation_of_features" in self._specified_features:
            corr_spearman = data.corr(method='spearman')
            corr_columns = list(corr_spearman.columns)
            corr_id = [data.columns.get_loc(n) for n in corr_columns]

        is_category = category_detection.category_detect(data)

        # STEP 2: column-level calculations
        # column_counter = -1
        for column_name in data:
            if self._verbose:
                print(f"==================Processing column {column_name} ... ==================\n")
            # column_counter += 1
            col = data[column_name]
            # dict: map feature name to content
            each_res = {}  # defaultdict(lambda: defaultdict())

            # if block updated on 6/26

            # old_metadata = dict(data.metadata.query((mbase.ALL_ELEMENTS, column_counter))),

            if 'semantic_types' in self._specified_features and is_category[column_name]:
                # rewrites old metadata
                # old_metadata = dict(data.metadata.query((mbase.ALL_ELEMENTS, column_counter)))
                old_metadata = metadata[column_name]
                temp_value = list(old_metadata["semantic_types"])
                if len(temp_value) == 2:
                    each_res["semantic_types"] = ('categorical', temp_value[-1])
                elif len(temp_value) == 1:
                    each_res["semantic_types"] = ('categorical', temp_value[-1])
                elif len(temp_value) == 3:
                    each_res["semantic_types"] = ('categorical', temp_value[-2], temp_value[-1])

            if (("spearman_correlation_of_features" in self._specified_features) and
                    (column_name in corr_columns)):
                stats_sp = corr_spearman[column_name].describe()
                each_res["spearman_correlation_of_features"] = {
                    'min': stats_sp['min'],
                    'max': stats_sp['max'],
                    'mean': stats_sp['mean'],
                    'median': stats_sp['50%'],
                    'std': stats_sp['std']
                }

            if (("spearman_correlation_of_features" in self._specified_features) and
                    (column_name in corr_columns)):
                stats_pr = corr_pearson[column_name].describe()
                each_res["pearson_correlation_of_features"] = {
                    'min': stats_pr['min'],
                    'max': stats_pr['max'],
                    'mean': stats_pr['mean'],
                    'median': stats_pr['50%'],
                    'std': stats_pr['std']
                }

            if col.dtype.kind in np.typecodes['AllInteger'] + 'uMmf':
                if "number_of_missing_values" in self._specified_features:
                    each_res["number_of_missing_values"] = pd.isnull(col).sum()
                if "ratio_of_missing_values" in self._specified_features:
                    each_res["ratio_of_missing_values"] = pd.isnull(col).sum() / col.size
                if "number_of_distinct_values" in self._specified_features:
                    each_res["number_of_distinct_values"] = col.nunique()
                if "ratio_of_distinct_values" in self._specified_features:
                    each_res["ratio_of_distinct_values"] = col.nunique() / float(col.size)

            if col.dtype.kind == 'b':
                if "most_common_raw_values" in self._specified_features:
                    fc_hih.compute_common_values(col.dropna().astype(str), each_res, self._topk)

            elif col.dtype.kind in np.typecodes['AllInteger'] + 'uf':
                # TODO: do the checks inside the function
                fc_hih.compute_numerics(col, each_res, self._specified_features)
                if "most_common_raw_values" in self._specified_features:
                    fc_hih.compute_common_values(col.dropna().astype(str), each_res, self._topk)
            else:
                # Need to compute str missing values before fillna
                if "number_of_missing_values" in self._specified_features:
                    each_res["number_of_missing_values"] = pd.isnull(col).sum()
                if "ratio_of_missing_values" in self._specified_features:
                    each_res["ratio_of_missing_values"] = pd.isnull(col).sum() / col.size

                col = col.astype(object).fillna('').astype(str)

                # compute_missing_space Must be put as the first one because it may change the
                # data content, see function def for details
                fc_lfh.compute_missing_space(col, each_res, self._specified_features)
                # fc_lfh.compute_filename(col, each_res)
                fc_lfh.compute_length_distinct(col, each_res, delimiter=self._token_delimiter,
                                               feature_list=self._specified_features)
                if "natural_language_of_feature" in self._specified_features:
                    fc_lfh.compute_lang(col, each_res)
                if "most_common_punctuations" in self._specified_features:
                    fc_lfh.compute_punctuation(col, each_res,
                                               weight_outlier=self._punctuation_outlier_weight)

                fc_hih.compute_numerics(col, each_res, self._specified_features)

                if "most_common_numeric_tokens" in self._specified_features:
                    fc_hih.compute_common_numeric_tokens(col, each_res, self._topk)
                if "most_common_alphanumeric_tokens" in self._specified_features:
                    fc_hih.compute_common_alphanumeric_tokens(col, each_res, self._topk)
                if "most_common_raw_values" in self._specified_features:
                    fc_hih.compute_common_values(col, each_res, self._topk)
                fc_hih.compute_common_tokens(col, each_res, self._topk, self._specified_features)
                if "numeric_char_density" in self._specified_features:
                    fc_hih.compute_numeric_density(col, each_res)
                fc_hih.compute_contain_numeric_values(col, each_res, self._specified_features)
                fc_hih.compute_common_tokens_by_puncs(col, each_res, self._topk,
                                                      self._specified_features)

            # update metadata for a specific column
            metadata[column_name] = dict(**metadata[column_name], **each_res)
            # metadata = metadata.update(prefix + [ALL_ELEMENTS, column_counter], each_res)

        return metadata
