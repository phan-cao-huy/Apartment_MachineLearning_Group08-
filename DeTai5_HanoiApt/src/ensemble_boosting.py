import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_i in range(n_features):
            X_column = X[:, feature_i]
            unique_vals = np.unique(X_column)
            if len(unique_vals) > 1:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                thresholds = np.concatenate([[unique_vals[0] - 1.0], thresholds, [unique_vals[-1] + 1.0]])
            else:
                thresholds = np.array([unique_vals[0] - 1.0, unique_vals[0] + 1.0])

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X_column >= threshold] = -1
                    else:
                        predictions[X_column < threshold] = -1

                    error = np.sum(sample_weights[y != predictions])
                    if error < min_error:
                        min_error = error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_idx = feature_i

        return min_error

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column >= self.threshold] = -1
        else:
            predictions[X_column < self.threshold] = -1
        return predictions


class AdaBoostClassifierCustom:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.weight_history_ = []
        self.train_error_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]
        w = np.full(n_samples, 1.0 / n_samples)

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.weight_history_ = [w.copy()]
        self.train_error_history_ = []

        F = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            error = stump.fit(X, y, w)
            error = np.clip(error, 1e-12, 1.0 - 1e-12)

            alpha = 0.5 * np.log((1.0 - error) / error)
            stump.alpha = alpha
            preds = stump.predict(X)

            w = w * np.exp(-alpha * y * preds)
            w /= np.sum(w)

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
            self.weight_history_.append(w.copy())

            F += alpha * preds
            train_pred = np.sign(F)
            train_pred[train_pred == 0] = 1
            train_err = np.mean(train_pred != y)
            self.train_error_history_.append(train_err)

            if error <= 1e-12:
                break

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.zeros(X.shape[0])
        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            F += alpha * stump.predict(X)
        return F

    def predict(self, X):
        F = self.decision_function(X)
        preds = np.sign(F)
        preds[preds == 0] = 1
        return preds


class RegressionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y) < 1e-12:
            return RegressionTreeNode(value=np.mean(y))

        best_gain = -1.0
        best_feat = None
        best_thresh = None
        parent_var = np.var(y) * n_samples

        for feat_idx in range(n_features):
            x_col = X[:, feat_idx]
            sorted_idx = np.argsort(x_col)
            x_sorted = x_col[sorted_idx]
            y_sorted = y[sorted_idx]

            valid_splits = np.where(x_sorted[:-1] != x_sorted[1:])[0]
            if len(valid_splits) == 0:
                continue

            cumsum_y = np.cumsum(y_sorted)
            cumsum_y2 = np.cumsum(y_sorted ** 2)
            total_y = cumsum_y[-1]
            total_y2 = cumsum_y2[-1]

            n_left = valid_splits + 1
            n_right = n_samples - n_left

            sum_left = cumsum_y[valid_splits]
            sum_right = total_y - sum_left

            sum2_left = cumsum_y2[valid_splits]
            sum2_right = total_y2 - sum2_left

            sse_left = sum2_left - (sum_left ** 2) / n_left
            sse_right = sum2_right - (sum_right ** 2) / n_right
            gain = parent_var - (sse_left + sse_right)

            max_idx = np.argmax(gain)
            if gain[max_idx] > best_gain:
                best_gain = gain[max_idx]
                best_feat = feat_idx
                split_pos = valid_splits[max_idx]
                best_thresh = 0.5 * (x_sorted[split_pos] + x_sorted[split_pos + 1])

        if best_gain <= 1e-12 or best_feat is None:
            return RegressionTreeNode(value=np.mean(y))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return RegressionTreeNode(value=np.mean(y))

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return RegressionTreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_node,
            right=right_node
        )

    def _predict_row(self, x, node):
        while node.value is None:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_row(row, self.root) for row in X])


class GradientBoostingRegressorCustom:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        subsample=1.0,
        loss='mse',
        alpha=0.9,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.loss = loss
        self.alpha = alpha
        self.random_state = random_state
        self.init_ = 0.0
        self.estimators_ = []
        self.train_loss_ = []

    def _compute_loss_and_pseudo_residuals(self, y, F):
        residual = y - F
        if self.loss == 'mse':
            loss_val = 0.5 * np.mean(residual ** 2)
            pseudo_res = residual
            return loss_val, pseudo_res, None
        elif self.loss == 'huber':
            abs_res = np.abs(residual)
            delta = np.quantile(abs_res, self.alpha)
            if delta < 1e-12:
                delta = 1e-12
            is_small = abs_res <= delta
            loss_val = np.mean(
                np.where(is_small, 0.5 * residual ** 2, delta * (abs_res - 0.5 * delta))
            )
            pseudo_res = np.where(is_small, residual, delta * np.sign(residual))
            return loss_val, pseudo_res, delta
        else:
            raise ValueError("loss must be 'mse' or 'huber'")

    def _update_leaf_values(self, node, X, residual, delta):
        if node.value is not None:
            if len(residual) > 0:
                med = np.median(residual)
                diff = residual - med
                node.value = med + np.mean(np.sign(diff) * np.minimum(np.abs(diff), delta))
            return
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = ~left_mask
        self._update_leaf_values(node.left, X[left_mask], residual[left_mask], delta)
        self._update_leaf_values(node.right, X[right_mask], residual[right_mask], delta)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        if self.loss == 'mse':
            self.init_ = np.mean(y)
        elif self.loss == 'huber':
            self.init_ = np.median(y)
        else:
            raise ValueError("loss must be 'mse' or 'huber'")

        F = np.full(n_samples, self.init_, dtype=float)
        self.estimators_ = []
        self.train_loss_ = []

        for _ in range(self.n_estimators):
            loss_val, pseudo_res, delta = self._compute_loss_and_pseudo_residuals(y, F)
            self.train_loss_.append(loss_val)

            if self.subsample < 1.0:
                sample_size = max(1, int(np.floor(self.subsample * n_samples)))
                idx = rng.choice(n_samples, size=sample_size, replace=False)
            else:
                idx = np.arange(n_samples)

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X[idx], pseudo_res[idx])

            if self.loss == 'huber':
                self._update_leaf_values(tree.root, X[idx], y[idx] - F[idx], delta)

            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F
