from skfeature.utility.entropy_estimators import *


def cost_based_lcsi(X, y, costs, alpha, n_selected_features):
    """
    This function implements the basic scoring criteria for linear combination of shannon information term.
    The scoring criteria is calculated based on the formula j_cmi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    y: {numpy array}, shape (n_samples,)
        input class labels
    n_selected_features: {int}
        number of features to select

    Output
    ------
    F: {numpy array}, shape: (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    info_term_scaler, cost_scaler = entropyd(y), costs.sum()
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMI = []

    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    # select the feature whose mutual information is the largest
    idx = np.argmax(t1/info_term_scaler - alpha*costs.values/cost_scaler)
    F.append(idx)
    J_CMI.append(t1[idx]/info_term_scaler - alpha*costs.iloc[idx]/cost_scaler)
    f_select = X[:, idx]

    if n_selected_features == 1:
        return np.array(F), np.array(J_CMI)

    while len(F) < n_selected_features:
        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
        j_cmi = -1E30
        beta = 1.0 / len(F)
        gamma = 1.0 / len(F)
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2[i] += midd(f_select, f)
                t3[i] += cmidd(f_select, f, y)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] - beta*t2[i] + gamma*t3[i]
                # calculate normalized and cost adjusted values for feature i
                t_adjusted = t/info_term_scaler - alpha*costs.iloc[i]/cost_scaler
                # record the largest j_cmi and the corresponding feature index
                if t_adjusted > j_cmi:
                    j_cmi = t_adjusted
                    idx = i

        F.append(idx)
        J_CMI.append(j_cmi)
        f_select = X[:, idx]

    return np.array(F), np.array(J_CMI)
