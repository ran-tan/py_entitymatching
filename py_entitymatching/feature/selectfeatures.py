"""
This module contains functions for selecting most relevant features.
"""

import six
import pandas as pd
# from math import ceil
# from scipy.stats import iqr
from py_entitymatching.utils.validation_helper import validate_object_type
from py_entitymatching.feature.attributeutils import get_attrs_to_project
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import GenericUnivariateSelect
from skfeature.utility.entropy_estimators import midd, cmidd
# from skfeature.utility.data_discretization import data_discretization
from skfeature.function.information_theoretical_based import MIFS, MRMR, CIFE, JMI, CMIM, ICAP, DISR, FCBF
from py_entitymatching.feature.discretizers import MDLPCDiscretizer
from py_entitymatching.feature.costbasedLCSI import cost_based_lcsi


def select_features_group_info(feature_table, table,
                               target_attr=None, exclude_attrs=None,
                               independent_attrs=None, parameter=2):
    # get attributes to project, validate parameters
    project_attrs = get_attrs_to_project(table=table,
                                         target_attr=target_attr,
                                         exclude_attrs=exclude_attrs)

    # project feature vectors into features:x and target:y
    x, y = table[project_attrs], table[target_attr]
    # discretize feature vectors
    discretizer = MDLPCDiscretizer()
    discretizer.fit_transform(x.values, y.values)

    feature_scores = []
    # group features by attribute and select the most relevant feature from each group
    for attr in independent_attrs:
        feature_group = \
            list(feature_table[feature_table.left_attribute == attr].feature_name.values)
        mutual_info = [midd(x[fn], y) for fn in feature_group]
        scored_features = list(zip(mutual_info, feature_group))
        max_rel = max(scored_features, key=lambda x: x[0])
        feature_scores.append(max_rel)

    feature_scores.sort(key=lambda x:x[0], reverse=True)

    feature_table_selected = pd.DataFrame(columns=feature_table.columns)
    for _, fn in feature_scores[:parameter]:
        ft = feature_table.loc[feature_table['feature_name'] == fn]
        feature_table_selected = pd.concat([feature_table_selected, ft])

    feature_table_selected.reset_index(inplace=True, drop=True)

    return feature_table_selected


def select_features_cost(feature_table, table, costs, alpha=0.0,
                         target_attr=None, exclude_attrs=None, parameter=2):
    # get attributes to project, validate parameters
    project_attrs = get_attrs_to_project(table=table,
                                         target_attr=target_attr,
                                         exclude_attrs=exclude_attrs)

    # project feature vectors into features:x and target:y
    x, y = table[project_attrs], table[target_attr]
    costs = costs.loc[project_attrs]
    # discretize feature vectors
    discretizer = MDLPCDiscretizer()
    discretizer.fit_transform(x.values, y.values)

    # fit and select most relevant features
    result = cost_based_lcsi(x.values, y.values, costs, alpha, parameter)

    # get selected features in feature_table
    feature_table_selected = pd.DataFrame(columns=feature_table.columns)
    for fn in x.columns[result]:
        ft = feature_table.loc[feature_table['feature_name'] == fn]
        feature_table_selected = pd.concat([feature_table_selected, ft])
    feature_table_selected.reset_index(inplace=True, drop=True)

    return feature_table_selected


def select_features_mi(feature_table, table,
                       target_attr=None, exclude_attrs=None,
                       mi_filter='MRMR', parameter=2):
    """
    This function will select a certain number of most relevant features according
    to the mutual information based filter specified by users.

    Specifically, this function will project the table into feature vectors (by
     excluding exclude_attrs and target_attr) and target values. It will then
     discretize the feature vector table and call one of the mutual information based
     filters provided by scikit-feature. The selected features is singled out
     in the given feature_table and returned.

    Args:
        feature_table (DataFrame): The pandas DataFrame which contains the metadata
            of all the features.
        table (DataFrame): The pandas DataFrames which contains all the features,
            keys, and target.
        target_attr (str): One attribute to be used as labels in the selection
            process, often with names `label` or `gold`.
        exclude_attrs (list): A list of attributes to be excluded from the table,
            these attributes will not be considered for feature selection.
        mi_filter (str): The filtering method specified by user, can be one of
            'MIFS', 'MRMR', 'CIFE', 'JMI', 'CMIM', 'ICAP', 'DISR','FCBF'.
            Defaults to be "MRMR".
        parameter (int): Number of features to be selected. Defaults to be 2.

    Returns:
        A pandas DataFrame of features selected as most relevant features, including
        all the metadata generated with the data.

    Raises:
        AssertionError: If `feature_table` is not of type pandas
            DataFrame.
        AssertionError: If `table` is not of type pandas
            DataFrame.
        AssertionError: If `mi_filter` is not of type
            str.
        AssertionError: If `parameter` is not of type
            int.
        AssertionError: If `mi_filter` is not in mi_filter_dict

    Examples:

        >>> import py_entitymatching as em
        >>> A = em.read_csv_metadata('path_to_csv_dir/table_A.csv', key='ID')
        >>> B = em.read_csv_metadata('path_to_csv_dir/table_B.csv', key='ID')
        >>> C = em.read_csv_metadata('path_to_csv_dir/table_C.csv', key='_id')
        >>> feature_table = get_features_for_matching(A, B, validate_inferred_attr_types=False)
        >>> F = extract_feature_vecs(C,
        >>>                          attrs_before=['_id', 'ltable.id', 'rtable.id'],
        >>>                          attrs_after=['gold'],
        >>>                          feature_table=feature_table)
        >>> x, scaler = scale_features(H,
        >>>                            exclude_attrs=['_id', 'ltable.id', 'rtable.id'],
        >>>                            scaling_method='MinMax')
        >>> feature_table_selected = select_features_mi(
        >>>     feature_table=feature_table, table=x,
        >>>     target_attr='gold', exclude_attrs=['_id', 'ltable.id', 'rtable.id'])

    See Also:
     :meth:`py_entitymatching.get_features_for_matching`,
     :meth:`py_entitymatching.extract_feature_vecs`,
     :meth:`py_entitymatching.scale_features`


    Note:
        The function applies only mutual information based feature selection methods.
        And returns only the metadata of selected features. To proceed, users need to
        used the selected features to extract feature vectors.

    """
    # Validate the input parameters
    # We expect the input object feature_table and table to be of type pandas DataFrame
    validate_object_type(feature_table, pd.DataFrame, 'Input feature_table')
    validate_object_type(table, pd.DataFrame, 'Input table')

    # validate parameters
    if not isinstance(mi_filter, six.string_types):
        raise AssertionError("Received wrong type of mutual information filter function")
    if not isinstance(parameter, int):
        raise AssertionError("Received wrong type of parameter")

    # get score function
    mi_filter_dict = _get_mi_funs()
    if mi_filter not in mi_filter_dict:
        raise AssertionError("Unknown score functions specified")

    if target_attr not in list(table.columns):
        raise AssertionError("Must specify the target attribute for feature selection")

    # get attributes to project, validate parameters
    project_attrs = get_attrs_to_project(table=table,
                                         target_attr=target_attr,
                                         exclude_attrs=exclude_attrs)

    mi_filter_fun = mi_filter_dict[mi_filter]

    # project feature vectors into features:x and target:y
    x, y = table[project_attrs], table[target_attr]
    # discretize feature vectors
    discretizer = MDLPCDiscretizer()
    discretizer.fit_transform(x.values, y.values)

    # fit and select most relevant features
    result = mi_filter_fun(x.values, y.values, n_selected_features=parameter)

    # get selected features in feature_table
    feature_table_selected = pd.DataFrame(columns=feature_table.columns)
    for fn in x.columns[result[0]]:
        ft = feature_table.loc[feature_table['feature_name'] == fn]
        feature_table_selected = pd.concat([feature_table_selected, ft])
    feature_table_selected.reset_index(inplace=True, drop=True)

    return feature_table_selected


def select_features_univariate(feature_table, table,
                               target_attr=None, exclude_attrs=None,
                               score='f_score', mode='k_best', parameter=2):
    """
    This function will select a certain number of most relevant features according
    to the criteria specified by users.

    Specifically, this function will project the table into feature vectors (by
     excluding exclude_attrs and target_attr) and target values. It will then call
     GenericUnivariateSelect provided by scikit-learn with specified scoring function,
     selection mode, and one specified parameter. The selected features is singled out
     in the given feature_table and returned.

    Args:
        feature_table (DataFrame): The pandas DataFrame which contains the metadata
            of all the features.
        table (DataFrame): The pandas DataFrames which contains all the features,
            keys, and target.
        target_attr (str): One attribute to be used as labels in the selection
            process, often with names `label` or `gold`.
        exclude_attrs (list): A list of attributes to be excluded from the table,
            these attributes will not be considered for feature selection.
        score (str): The scoring method specified by user, can be one of
            "chi_square", "f_score", "mutual_info". Defaults to be "f_score".
        mode (str): The selection mode specified by user, can be one of
            "percentile", "k_best", "fpr", "fdr", "fwe". Defaults to be "k_best".
        parameter (int or float): The parameter specified by the user according to
            the chosen mode.  Defaults to be 2.

    Returns:
        A pandas DataFrame of features selected as most relevant features, including
        all the metadata generated with the data.

    Raises:
        AssertionError: If `feature_table` is not of type pandas
            DataFrame.
        AssertionError: If `table` is not of type pandas
            DataFrame.
        AssertionError: If `score` is not of type
            str.
        AssertionError: If `mode` is not of type
            str.
        AssertionError: If `parameter` is not of type
            int or float.
        AssertionError: If `score` is not in score_dict
        AssertionError: If `mode` is not in mode_list

    Examples:

        >>> import py_entitymatching as em
        >>> A = em.read_csv_metadata('path_to_csv_dir/table_A.csv', key='ID')
        >>> B = em.read_csv_metadata('path_to_csv_dir/table_B.csv', key='ID')
        >>> C = em.read_csv_metadata('path_to_csv_dir/table_C.csv', key='_id')
        >>> feature_table = get_features_for_matching(A, B, validate_inferred_attr_types=False)
        >>> F = extract_feature_vecs(C,
        >>>                          attrs_before=['_id', 'ltable.id', 'rtable.id'],
        >>>                          attrs_after=['gold'],
        >>>                          feature_table=feature_table)
        >>> x, scaler = scale_features(H,
        >>>                            exclude_attrs=['_id', 'ltable.id', 'rtable.id'],
        >>>                            scaling_method='MinMax')
        >>> feature_table_selected = select_features_univariate(
        >>>     feature_table=feature_table, table=x,
        >>>     target_attr='gold', exclude_attrs=['_id', 'ltable.id', 'rtable.id'],
        >>>     score='f_score', mode='k_best', parameter=2)

    See Also:
     :meth:`py_entitymatching.get_features_for_matching`,
     :meth:`py_entitymatching.extract_feature_vecs`,
     :meth:`py_entitymatching.scale_features`


    Note:
        The function applies only univariate feature selection methods. And returns
        only the metadata of selected features. To proceed, users need to used the
        selected features to extract feature vectors.

    """
    # Validate the input parameters
    # We expect the input object feature_table and table to be of type pandas DataFrame
    validate_object_type(feature_table, pd.DataFrame, 'Input feature_table')
    validate_object_type(table, pd.DataFrame, 'Input table')

    # validate parameters
    if not isinstance(score, six.string_types):
        raise AssertionError("Received wrong type of score function")
    if not isinstance(mode, six.string_types):
        raise AssertionError("Received wrong type of mode function")
    if not (isinstance(parameter, int) or isinstance(parameter, float)):
        raise AssertionError("Received wrong type of parameter")

    # get score function
    score_dict = _get_score_funs()
    # get mode names allowed
    mode_list = _get_mode_names()
    if score not in score_dict:
        raise AssertionError("Unknown score functions specified")
    if mode not in mode_list:
        raise AssertionError("Unknown mode specified")

    if target_attr not in list(table.columns):
        raise AssertionError("Must specify the target attribute for feature selection")

    # get attributes to project, validate parameters
    project_attrs = get_attrs_to_project(table=table,
                                         target_attr=target_attr,
                                         exclude_attrs=exclude_attrs)

    score_fun = score_dict[score]
    # initialize selector with the given specification
    selector = GenericUnivariateSelect(score_func=score_fun,
                                       mode=mode,
                                       param=parameter)

    # project feature vectors into features:x and target:y
    x, y = table[project_attrs], table[target_attr]

    # fit and select most relevant features
    selector.fit(x, y)
    idx = selector.get_support(indices=True)

    # get selected features in feature_table
    feature_names_selected = x.columns[idx]
    feature_table_selected = feature_table.loc[feature_table['feature_name'].isin(feature_names_selected)]
    feature_table_selected.reset_index(inplace=True, drop=True)

    return feature_table_selected


def _get_score_funs():
    """
    This function returns the score functions specified by score.

    """
    # Get all the score names
    score_names = ['chi_square',
                   'f_score',
                   'mutual_info']
    # Get all the score functions
    score_funs = [chi2,
                  f_classif,
                  mutual_info_classif]
    # Return a dictionary with the scores names as the key and the actual
    # score functions as values.
    return dict(zip(score_names, score_funs))


def _get_mode_names():
    # Get names of all modes allowed
    return ['percentile', 'k_best', 'fpr', 'fdr', 'fwe']


def _get_mi_funs():
    """
    This function returns the mutual information based filter functions
    specified by the given filter.

    """
    # Get all the mi filter names
    mi_names = ['MIFS',
                'MRMR',
                'CIFE',
                'JMI',
                'CMIM',
                'ICAP',
                'DISR',
                'FCBF']
    # Get all the mi filter functions
    mi_funs = [MIFS.mifs,
               MRMR.mrmr,
               CIFE.cife,
               JMI.jmi,
               CMIM.cmim,
               ICAP.icap,
               DISR.disr,
               FCBF.fcbf]
    # Return a dictionary with the mi filter names as the key and
    # the actual filter functions as values.
    return dict(zip(mi_names, mi_funs))

# # Deprecated discretizer function
# def _discretize(array):
#     # Get the shape of the array
#     n_sample, n_feature = array.shape
#     # Apply Freedman-Diaconis' rule (no assumption on the distribution)
#     # to estimate the number of bins needed for discretization
#     bins = ceil(n_sample ** (1 / 3.0) / (2.0 * iqr(array)))
#     # Return discretized array
#     return data_discretization(array, bins)
