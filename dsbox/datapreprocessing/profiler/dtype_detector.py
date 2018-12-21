import typing

import pandas as pd

import logging

MetaData_T = typing.Dict[str, typing.Any]

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def detect_numbers(inputs_in: pd.DataFrame, metadata: MetaData_T):
    inputs = inputs_in.copy(deep=True)
    # _logger = logging.getLogger(__name__)
    for col, name in enumerate(inputs.columns.values):
        if name not in metadata:
            meta_col = {
                "structural_type": set(),
                "semantic_types": set()
            }
        else:
            meta_col = metadata[name]

        temp_col = inputs.loc[:, name]
        dtype = pd.DataFrame(temp_col.dropna().astype(str).str.isnumeric().value_counts())

        # if there is already a data type, see if that is equal to what we identified,
        # else update corner case : Integer type, could be a categorical Arrtribute detect integers
        # and update metadata
        if True in dtype.index and dtype.loc[True][0] == temp_col.dropna().shape[0]:
            meta_col["structural_type"].add(type(10),)
            meta_col['semantic_types'].add('int')
        # detetct Float and update metadata
        else:
            dtype = pd.DataFrame(temp_col.dropna().apply(isfloat).value_counts())
            if True in dtype.index and dtype.loc[True][0] == temp_col.dropna().shape[0]:
                meta_col["structural_type"].add(type(10.0),)
                meta_col['semantic_types'].add('float')

        metadata[name] = meta_col
        # inputs.metadata = inputs.metadata.update((mbase.ALL_ELEMENTS, col), old_metadata)

    return metadata


