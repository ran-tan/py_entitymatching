from skfeature.utility.entropy_estimators import *


def cost_based_lcsi(X, y, costs, alpha, fade_rate, n_selected_features):
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

    fade = fade_rate
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
                t_adjusted = t/info_term_scaler - alpha * fade * costs.iloc[i]/cost_scaler
                # record the largest j_cmi and the corresponding feature index
                if t_adjusted > j_cmi:
                    j_cmi = t_adjusted
                    idx = i

        F.append(idx)
        J_CMI.append(j_cmi)
        f_select = X[:, idx]
        fade *= fade_rate

    return np.array(F), np.array(J_CMI)


def cost_based_cmim(X, y, costs, alpha, fade_rate, n_selected_features):
    """
    This function implements the CMIM feature selection.
    The scoring criteria is calculated based on the formula j_cmim=I(f;y)-max_j(I(fj;f)-I(fj;f|y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y: {numpy array}, shape (n_samples,)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMIM: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    info_term_scaler, cost_scaler = entropyd(y), costs.sum()
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMIM = []

    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)

    # max stores max(I(fj;f)-I(fj;f|y)) for each feature f
    # we assign an extreme small value to max[i] ito make it is smaller than possible value of max(I(fj;f)-I(fj;f|y))
    max = -1E30 * np.ones(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    # select the feature whose mutual information is the largest
    idx = np.argmax(t1)
    F.append(idx)
    J_CMIM.append(t1[idx])
    f_select = X[:, idx]

    fade = fade_rate
    while len(F) < n_selected_features:
        # we assign an extreme small value to j_cmim to ensure it is smaller than all possible values of j_cmim
        j_cmim = -1E30
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2 = midd(f_select, f)
                t3 = cmidd(f_select, f, y)
                if t2-t3 > max[i]:
                        max[i] = t2-t3
                # calculate j_cmim for feature i (not in F)
                t = t1[i] - max[i]
                # calculate normalized and cost adjusted values for feature i
                t_adjusted = t / info_term_scaler - alpha * fade * costs.iloc[i] / cost_scaler
                # record the largest j_cmim and the corresponding feature index
                if t_adjusted > j_cmim:
                    j_cmim = t_adjusted
                    idx = i
        F.append(idx)
        J_CMIM.append(j_cmim)
        f_select = X[:, idx]
        fade *= fade_rate

    return np.array(F), np.array(J_CMIM)