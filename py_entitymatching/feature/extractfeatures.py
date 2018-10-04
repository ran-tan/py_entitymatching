"""
This module contains functions to extract features using a feature table.
"""
import logging

import multiprocessing
import time
import os

import pandas as pd
import pyprind
import tempfile

from cloudpickle import cloudpickle
from joblib import Parallel
from joblib import delayed

import py_entitymatching.catalog.catalog_manager as cm
import py_entitymatching.utils.catalog_helper as ch
import py_entitymatching.utils.generic_helper as gh
from py_entitymatching.io.pickles import save_object, load_object
from py_entitymatching.utils.validation_helper import validate_object_type

logger = logging.getLogger(__name__)


def extract_feature_vecs(candset, attrs_before=None, feature_table=None,
                         attrs_after=None, verbose=False, get_cost=False,
                         show_progress=True, n_jobs=1):
    """
    This function extracts feature vectors from a DataFrame (typically a
    labeled candidate set).

    Specifically, this function uses feature
    table, ltable and rtable (that is present in the `candset`'s
    metadata) to extract feature vectors.

    Args:
        candset (DataFrame): The input candidate set for which the features
            vectors should be extracted.
        attrs_before (list): The list of attributes from the input candset,
            that should be added before the feature vectors (defaults to None).
        feature_table (DataFrame): A DataFrame containing a list of
            features that should be used to compute the feature vectors (
            defaults to None).
        attrs_after (list): The list of attributes from the input candset
            that should be added after the feature vectors (defaults to None).
        verbose (boolean): A flag to indicate whether the debug information
            should be displayed (defaults to False).
        get_cost (boolean): A flag to indicate whether to return the cost of
            extracting each feature vector (defaults to False).
        show_progress (boolean): A flag to indicate whether the progress of
            extracting feature vectors must be displayed (defaults to True).
        n_jobs (integer): A integer to indicate the number of parallel jobs
            (defaults to 1).


    Returns:
        A pandas DataFrame containing feature vectors.

        The DataFrame will have metadata ltable and rtable, pointing
        to the same ltable and rtable as the input candset.

        Also, the output
        DataFrame will have three columns: key, foreign key ltable, foreign
        key rtable copied from input candset to the output DataFrame. These
        three columns precede the columns mentioned in `attrs_before`.



    Raises:
        AssertionError: If `candset` is not of type pandas
            DataFrame.
        AssertionError: If `attrs_before` has attributes that
            are not present in the input candset.
        AssertionError: If `attrs_after` has attribtues that
            are not present in the input candset.
        AssertionError: If `feature_table` is set to None.


    Examples:
        >>> import py_entitymatching as em
        >>> A = em.read_csv_metadata('path_to_csv_dir/table_A.csv', key='ID')
        >>> B = em.read_csv_metadata('path_to_csv_dir/table_B.csv', key='ID')
        >>> match_f = em.get_features_for_matching(A, B)
        >>> # G is the labeled dataframe which should be converted into feature vectors
        >>> H = em.extract_feature_vecs(G, features=match_f, attrs_before=['title'], attrs_after=['gold_labels'])


    """
    # Validate input parameters

    # # We expect the input candset to be of type pandas DataFrame.
    validate_object_type(candset, pd.DataFrame, error_prefix='Input cand.set')

    # # If the attrs_before is given, Check if the attrs_before are present in
    # the input candset
    if attrs_before != None:
        if not ch.check_attrs_present(candset, attrs_before):
            logger.error(
                'The attributes mentioned in attrs_before is not present '
                'in the input table')
            raise AssertionError(
                'The attributes mentioned in attrs_before is not present '
                'in the input table')

    # # If the attrs_after is given, Check if the attrs_after are present in
    # the input candset
    if attrs_after != None:
        if not ch.check_attrs_present(candset, attrs_after):
            logger.error(
                'The attributes mentioned in attrs_after is not present '
                'in the input table')
            raise AssertionError(
                'The attributes mentioned in attrs_after is not present '
                'in the input table')

    # We expect the feature table to be a valid object
    if feature_table is None:
        logger.error('Feature table cannot be null')
        raise AssertionError('The feature table cannot be null')

    # Do metadata checking
    # # Mention what metadata is required to the user
    ch.log_info(logger, 'Required metadata: cand.set key, fk ltable, '
                        'fk rtable, '
                        'ltable, rtable, ltable key, rtable key', verbose)

    # # Get metadata
    ch.log_info(logger, 'Getting metadata from catalog', verbose)

    key, fk_ltable, fk_rtable, ltable, rtable, l_key, r_key = \
        cm.get_metadata_for_candset(
            candset, logger, verbose)

    # # Validate metadata
    ch.log_info(logger, 'Validating metadata', verbose)
    cm._validate_metadata_for_candset(candset, key, fk_ltable, fk_rtable,
                                      ltable, rtable, l_key, r_key,
                                      logger, verbose)

    # Extract features



    # id_list = [(row[fk_ltable], row[fk_rtable]) for i, row in
    #            candset.iterrows()]
    # id_list = [tuple(tup) for tup in candset[[fk_ltable, fk_rtable]].values]
    if feature_table.empty:
        feature_vectors = pd.DataFrame()
        feature_costs = pd.DataFrame()
    else:
        # # Set index for convenience
        l_df = ltable.set_index(l_key, drop=False)
        r_df = rtable.set_index(r_key, drop=False)

        # # Apply feature functions
        ch.log_info(logger, 'Applying feature functions', verbose)
        col_names = list(candset.columns)
        fk_ltable_idx = col_names.index(fk_ltable)
        fk_rtable_idx = col_names.index(fk_rtable)

        n_procs = get_num_procs(n_jobs, len(candset))

        c_splits = pd.np.array_split(candset, n_procs)

        pickled_obj = cloudpickle.dumps(feature_table)

        feat_vals_by_splits = Parallel(n_jobs=n_procs)(delayed(get_feature_vals_by_cand_split)(pickled_obj,
                                                                                               fk_ltable_idx,
                                                                                               fk_rtable_idx,
                                                                                               l_df, r_df,
                                                                                               c_splits[i],
                                                                                               show_progress and i == len(
                                                                                                   c_splits) - 1)
                                                       for i in range(len(c_splits)))
        feat_vals, costs = zip(*feat_vals_by_splits)
        feature_vectors = pd.concat(feat_vals, axis=0, ignore_index=True)
        feature_costs = pd.concat(costs, axis=0, ignore_index=True).sum()

    ch.log_info(logger, 'Constructing output table', verbose)

    # # Insert attrs_before
    if attrs_before:
        if not isinstance(attrs_before, list):
            attrs_before = [attrs_before]
        attrs_before = gh.list_diff(attrs_before, [key, fk_ltable, fk_rtable])
        attrs_before.reverse()
        for a in attrs_before:
            feature_vectors.insert(0, a, candset[a])

    # # Insert keys
    feature_vectors.insert(0, fk_rtable, candset[fk_rtable])
    feature_vectors.insert(0, fk_ltable, candset[fk_ltable])
    feature_vectors.insert(0, key, candset[key])

    # # insert attrs after
    if attrs_after:
        if not isinstance(attrs_after, list):
            attrs_after = [attrs_after]
        attrs_after = gh.list_diff(attrs_after, [key, fk_ltable, fk_rtable])
        attrs_after.reverse()
        col_pos = len(feature_vectors.columns)
        for a in attrs_after:
            feature_vectors.insert(col_pos, a, candset[a])
            col_pos += 1

    # Reset the index
    # feature_vectors.reset_index(inplace=True, drop=True)

    # # Update the catalog
    cm.init_properties(feature_vectors)
    cm.copy_properties(candset, feature_vectors)

    if get_cost:
        return feature_vectors, feature_costs
    # Finally, return the feature vectors
    else:
        return feature_vectors


def get_feature_vals_by_cand_split(pickled_obj, fk_ltable_idx, fk_rtable_idx, l_df, r_df, candsplit, show_progress):
    feature_table = cloudpickle.loads(pickled_obj)
    if show_progress:
        prog_bar = pyprind.ProgBar(len(feature_table))

    fk_ltable_vals = candsplit.iloc[:, fk_ltable_idx]
    fk_rtable_vals = candsplit.iloc[:, fk_rtable_idx]

    ltable_tuples = [l_df.loc[fk_ltable_val] for fk_ltable_val in fk_ltable_vals]
    rtable_tuples = [r_df.loc[fk_rtable_val] for fk_rtable_val in fk_rtable_vals]

    feat_vals = []
    costs = []
    for row in feature_table.itertuples(index=False):
        if show_progress:
            prog_bar.update()

        meta_feat = {col: val for col, val in zip(feature_table.columns, row)}

        feat_val, cost = apply_feat_fns(ltable_tuples, rtable_tuples, meta_feat)
        feat_vals.append(feat_val)
        costs.append(cost)

    return pd.concat(feat_vals, axis=1), pd.concat(costs, axis=1)


def apply_feat_fns(ltable_tuples, rtable_tuples, meta_feat):
    name = meta_feat['feature_name']
    func = meta_feat['function']
    s_time = time.process_time()
    feat_vals = [func(l_tuple, r_tuple)
                 for l_tuple, r_tuple in zip(ltable_tuples, rtable_tuples)]
    e_time = time.process_time()

    feat_vector = pd.DataFrame.from_dict({name: feat_vals})
    cost = pd.DataFrame.from_dict({name: [e_time - s_time]})

    return feat_vector, cost


def get_num_procs(n_jobs, min_procs):
    # determine number of processes to launch parallely
    n_cpus = multiprocessing.cpu_count()
    n_procs = n_jobs
    if n_jobs < 0:
        n_procs = n_cpus + 1 + n_jobs
    # cannot launch less than min_procs to safeguard against small tables
    return min(n_procs, min_procs)
