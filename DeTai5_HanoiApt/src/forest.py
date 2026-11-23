"""
forest.py - A.2: Bagging và rừng ngẫu nhiên (Random Forest) kèm sai số out-of-bag.

Nội dung
--------
* RandomForestRegressor / RandomForestClassifier
    - mỗi cây học trên một mẫu bootstrap;
    - tại MỖI NÚT chỉ xét ngẫu nhiên `max_features` đặc trưng (lấy mẫu đặc trưng,
      làm giảm tương quan rho giữa các cây -> giảm Var theo công thức ở B.2).
* BaggingRegressor / BaggingClassifier
    - chính là rừng ngẫu nhiên với max_features=None (xét mọi đặc trưng).
* Ước lượng out-of-bag (OOB): mỗi mẫu được dự đoán chỉ bởi các cây KHÔNG thấy
  mẫu đó khi huấn luyện -> ước lượng sai số tổng quát hóa mà không cần tập kiểm tra.
    - oob_score_, oob_error_, oob_prediction_ / oob_decision_function_
    - oob_error_curve_[k-1] = sai số OOB khi chỉ dùng k cây đầu tiên
      (dùng cho bài tập C.4: sai số OOB theo số cây).

Ví dụ
-----
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                               oob_score=True, random_state=0).fit(X, y)
    rf.oob_error_          # MSE out-of-bag
    rf.oob_score_          # R^2 out-of-bag
    rf.predict(X_new)
"""
from __future__ import annotations

import numpy as np

from cart import DecisionTreeClassifier, DecisionTreeRegressor


def _fit_one_tree(tree_cls, params, X, y, m, bootstrap, seed):
    """Huấn luyện một cây trên một mẫu bootstrap; trả về (cây, chỉ số OOB)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if bootstrap:
        idx = rng.integers(0, n, size=m)           # lấy có hoàn lại
        in_bag = np.zeros(n, dtype=bool)
        in_bag[idx] = True
        oob_idx = np.flatnonzero(~in_bag)
    else:
        idx = np.arange(n)
        oob_idx = np.empty(0, dtype=np.int64)
    tree = tree_cls(random_state=rng, **params)
    tree.fit(X[idx], y[idx])
    return tree, oob_idx


class _BaseForest:
    _is_classifier = False
    _tree_cls = None
    _default_criterion = None

    def __init__(self, n_estimators=100, criterion=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="auto",
                 ccp_alpha=0.0, bootstrap=True, max_samples=None,
                 oob_score=False, n_jobs=None, random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _resolve_max_features(self):
        if self.max_features == "auto":
            # Khuyến nghị của Breiman: sqrt(d) cho phân loại, d/3 cho hồi quy
            return "sqrt" if self._is_classifier else 1.0 / 3.0
        return self.max_features

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[0]
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True cần bootstrap=True")

        if self._is_classifier:
            self.classes_, y_enc = np.unique(y, return_inverse=True)
        else:
            y_enc = y.astype(np.float64)

        ms = self.max_samples
        m = n if ms is None else (int(ms * n) if isinstance(ms, float) else int(ms))
        m = max(1, m)

        params = dict(
            criterion=self.criterion or self._default_criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self._resolve_max_features(),
            ccp_alpha=self.ccp_alpha,
        )
        # Mỗi cây nhận một seed riêng -> kết quả không phụ thuộc n_jobs
        seeds = np.random.default_rng(self.random_state).integers(
            0, 2**32 - 1, size=self.n_estimators)

        if self.n_jobs in (None, 1):
            results = [_fit_one_tree(self._tree_cls, params, X, y_enc, m, self.bootstrap, s)
                       for s in seeds]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_one_tree)(self._tree_cls, params, X, y_enc, m, self.bootstrap, s)
                for s in seeds)

        self.estimators_ = [r[0] for r in results]
        self._oob_idx = [r[1] for r in results]
        self.n_features_in_ = X.shape[1]
        if self.oob_score:
            self._compute_oob(X, y_enc)
        return self

    # ------------------------------------------------------------------
    def _tree_output(self, tree, X):
        """Dự đoán của một cây: giá trị (hồi quy) hoặc xác suất đủ K lớp (phân loại)."""
        if not self._is_classifier:
            return tree.predict(X)
        P = np.zeros((X.shape[0], self.classes_.size))
        P[:, tree.classes_] = tree.predict_proba(X)   # cây có thể thiếu lớp nào đó
        return P

    def _compute_oob(self, X, y_enc):
        n = X.shape[0]
        if self._is_classifier:
            acc = np.zeros((n, self.classes_.size))
        else:
            acc = np.zeros(n)
        cnt = np.zeros(n)
        curve = []
        for tree, oob in zip(self.estimators_, self._oob_idx):
            if oob.size:
                acc[oob] += self._tree_output(tree, X[oob])
                cnt[oob] += 1
            cov = cnt > 0
            if not cov.any():
                curve.append(np.nan)
            elif self._is_classifier:
                curve.append(float(np.mean(np.argmax(acc[cov], axis=1) != y_enc[cov])))
            else:
                curve.append(float(np.mean((y_enc[cov] - acc[cov] / cnt[cov]) ** 2)))
        self.oob_error_curve_ = np.array(curve)
        self.oob_error_ = float(curve[-1])
        self.oob_coverage_ = float(np.mean(cnt > 0))   # tỷ lệ mẫu có ít nhất 1 cây OOB

        cov = cnt > 0
        if self._is_classifier:
            dec = np.full((n, self.classes_.size), np.nan)
            dec[cov] = acc[cov] / cnt[cov, None]
            self.oob_decision_function_ = dec
            self.oob_score_ = 1.0 - self.oob_error_            # độ chính xác OOB
        else:
            pred = np.full(n, np.nan)
            pred[cov] = acc[cov] / cnt[cov]
            self.oob_prediction_ = pred
            sse = np.sum((y_enc[cov] - pred[cov]) ** 2)
            sst = np.sum((y_enc[cov] - y_enc[cov].mean()) ** 2)
            self.oob_score_ = float(1.0 - sse / sst)            # R^2 OOB

    # ------------------------------------------------------------------
    @property
    def feature_importances_(self):
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        s = imp.sum()
        return imp / s if s > 0 else imp


class RandomForestRegressor(_BaseForest):
    _tree_cls = DecisionTreeRegressor
    _default_criterion = "squared_error"

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)

    def score(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        return float(1.0 - np.sum((y - self.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2))


class RandomForestClassifier(_BaseForest):
    _is_classifier = True
    _tree_cls = DecisionTreeClassifier
    _default_criterion = "gini"

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.mean([self._tree_output(t, X) for t in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# --------------------------------------------------------------------------
# Bagging = rừng ngẫu nhiên không lấy mẫu đặc trưng (mọi đặc trưng ở mỗi nút)
# --------------------------------------------------------------------------
class BaggingRegressor(RandomForestRegressor):
    def __init__(self, n_estimators=100, **kwargs):
        kwargs.pop("max_features", None)
        super().__init__(n_estimators=n_estimators, max_features=None, **kwargs)


class BaggingClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=100, **kwargs):
        kwargs.pop("max_features", None)
        super().__init__(n_estimators=n_estimators, max_features=None, **kwargs)
