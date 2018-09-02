import numpy as np
from math import log
from pyitlib import discrete_random_variable as drv
from sklearn.base import TransformerMixin


class MDLPCDiscretizer(TransformerMixin):
    def __init__(self, **kwargs):
        self.features = kwargs.get('features', None)
        self.labels = kwargs.get('labels', None)
        self.column_indices = kwargs.get('column_indices', None)
        self.cuts = {}
        self.boundaries = []
        self.bin_descriptions = {}

    #######################################
    # Interfaces
    #######################################

    def fit(self, X, y, **kwargs):
        self.features = X
        self.labels = y
        self.column_indices = kwargs.get('column_indices', np.arange(self.features.shape[1]))
        # initialize feature bins cut points
        self.cuts = {idx: [] for idx in self.column_indices}

        self.all_features_boundary_points()
        self.all_features_accepted_cutpoints()
        self.generate_bin_descriptions()

        return self

    def transform(self, X, inplace=False):
        return self.apply_cutpoints(X) if inplace \
            else self.apply_cutpoints(X.copy())

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X, inplace=True)

    #######################################
    # Initialize boundary points
    #######################################

    def feature_boundary_points(self, feature):
        sample_num = feature.size
        joint = np.column_stack((feature, self.labels))
        sorted_joint = joint[joint[:, 0].argsort()]
        feature, label = sorted_joint[:, 0], sorted_joint[:, 1]
        unique_vals = np.unique(feature)

        # Find if when feature changes there are different class values
        boundaries = []
        for i in range(1, unique_vals.size):  # By definition first unique value cannot be a boundary
            pre_val_idx = np.where(feature == unique_vals[i-1])
            cur_val_idx = np.where(feature == unique_vals[i])
            merged_classes = np.union1d(label[pre_val_idx], label[cur_val_idx])
            if merged_classes.size > 1:
                boundaries.append((unique_vals[i-1] + unique_vals[i]) / 2)

        padding = np.array([np.nan] * (sample_num - len(boundaries)))

        return np.concatenate((np.array(boundaries), padding))

    def all_features_boundary_points(self):
        boundaries = np.apply_along_axis(
            self.feature_boundary_points, 0,
            self.features[:, self.column_indices])
        mask = np.all(np.isnan(boundaries), axis=1)

        self.boundaries = boundaries[~mask]

    #######################################
    # Apply MDLPC to get cut points
    #######################################
    @staticmethod
    def cut_point_information_gain(X, y, cut_point):
        left_mask = X <= cut_point
        right_mask = X > cut_point
        (N, N_left, N_right) = (len(X), left_mask.sum(), right_mask.sum())

        gain = drv.entropy(y) - \
               (N_left / N) * drv.entropy(y[left_mask]) - \
               (N_right / N) * drv.entropy(y[right_mask])

        return gain

    @staticmethod
    def MDLPC_criterion(X, y, cut_point):
        left_mask = X <= cut_point
        right_mask = X > cut_point
        N, k = len(X), len(np.unique(y))
        k_left, k_right = len(np.unique(y[left_mask])), len(np.unique(y[right_mask]))

        delta = log(3 ** k, 2) - (k * drv.entropy(y)) + \
                (k_left * drv.entropy(y[left_mask])) + \
                (k_right * drv.entropy(y[right_mask]))

        gain_threshold = (log(N - 1, 2) + delta) / N

        return gain_threshold

    def best_cut_point(self, feature, label, feature_idx):
        mask = np.logical_and((self.boundaries[:, feature_idx] > feature.min()),
                              (self.boundaries[:, feature_idx] < feature.max()))
        candidates = np.unique(self.boundaries[:, feature_idx][mask])

        if candidates.size == 0:
            return None

        gains = [(cut, self.cut_point_information_gain(feature, label, cut_point=cut))
                 for cut in candidates]
        max_gain = max(gains, key=lambda x: x[1])[0]

        return max_gain

    def single_feature_accepted_cutpoints(self, x, y, feature_idx):
        # Delete missing data
        mask = np.isnan(x)
        x = x[~mask]
        y = y[~mask]

        # stop if feature value is constant or null
        if len(np.unique(x)) < 2:
            return
        # determine whether to cut and where to cut
        cut_candidate = self.best_cut_point(x, y, feature_idx)
        if cut_candidate is None:
            return

        cut_point_gain = self.cut_point_information_gain(x, y, cut_candidate)
        gain_threshold = self.MDLPC_criterion(x, y, cut_candidate)

        if cut_point_gain > gain_threshold:
            # partition masks
            left_mask = x <= cut_candidate
            right_mask = x > cut_candidate
            # now we have two new partitions that need to be examined
            left_partition = x[left_mask]
            right_partition = x[right_mask]
            if (left_partition.size == 0) or (right_partition.size == 0):
                return
            self.cuts[feature_idx] += [cut_candidate]  # accept partition
            self.single_feature_accepted_cutpoints(left_partition, y[left_mask], feature_idx)
            self.single_feature_accepted_cutpoints(right_partition, y[right_mask], feature_idx)
            self.cuts[feature_idx] = sorted(self.cuts[feature_idx])

    def all_features_accepted_cutpoints(self):
        for idx in self.column_indices:
            self.single_feature_accepted_cutpoints(
                x=self.features[:, idx],
                y=self.labels,
                feature_idx=idx)

    #######################################
    # transform & discretize features
    #######################################

    def apply_cutpoints(self, data):
        for col_idx in self.column_indices:
            if len(self.cuts[col_idx]) == 0:
                data[:, col_idx] = 0
            else:
                cuts = [-np.inf] + self.cuts[col_idx] + [np.inf]
                discretized_col = np.digitize(x=data[:, col_idx], bins=cuts, right=False).astype('float') - 1
                discretized_col[np.isnan(data[:, col_idx])] = np.nan
                data[:, col_idx] = discretized_col

        return data

    def generate_bin_descriptions(self):
        bin_label_collection = {}
        for col_idx in self.column_indices:
            if len(self.cuts[col_idx]) == 0:
                bin_label_collection[col_idx] = ['All']
            else:
                cuts = [-np.inf] + self.cuts[col_idx] + [np.inf]
                start_bin_indices = range(0, len(cuts) - 1)
                bin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i+1])) for i in start_bin_indices]
                bin_label_collection[col_idx] = bin_labels
                self.bin_descriptions[col_idx] = {i: bin_labels[i] for i in range(len(bin_labels))}
