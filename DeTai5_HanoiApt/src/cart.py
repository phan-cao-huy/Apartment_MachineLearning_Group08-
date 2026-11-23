"""
cart.py - A.1: Cài đặt CART (Classification And Regression Trees) bằng NumPy.

Nội dung
--------
* DecisionTreeClassifier : phân loại, tiêu chí "gini" hoặc "entropy".
* DecisionTreeRegressor  : hồi quy, tiêu chí sai số bình phương ("squared_error").
* Cắt tỉa theo độ phức tạp chi phí (cost-complexity pruning, Breiman 1984):
    - tree.cost_complexity_pruning_path()  -> (ccp_alphas, impurities)
    - tree.prune(alpha)                    -> cây đã cắt tỉa (bản sao)
    - DecisionTree*(ccp_alpha=alpha)       -> cắt tỉa ngay sau khi fit
* Hỗ trợ sample_weight (dùng lại cho AdaBoost / boosting ở A.3, A.4) và
  max_features (lấy mẫu đặc trưng tại mỗi nút, dùng cho rừng ngẫu nhiên ở A.2).

Ý tưởng cài đặt
---------------
Mọi tiêu chí được viết dưới dạng "vector thống kê" cộng được:
  * phân loại: s = tổng trọng số theo từng lớp            (K phần tử)
  * hồi quy  : s = (sum w, sum w*y, sum w*y^2)            (3 phần tử)
Nhờ đó, với mỗi đặc trưng chỉ cần sắp xếp một lần rồi dùng cumsum để có
thống kê của nhánh trái cho MỌI ngưỡng cùng lúc (độ phức tạp O(n log n)).

Cây lưu dưới dạng mảng phẳng (feature_, threshold_, left_, right_, ...);
nút lá có left_ == right_ == -1. Id nút con luôn lớn hơn id nút cha.
"""
from __future__ import annotations

import copy
import heapq

import numpy as np

LEAF = -1
_EPS = 1e-12


# --------------------------------------------------------------------------
# Hàm độ tạp chất (tính theo vector tỷ lệ lớp p, trục cuối là trục lớp)
# --------------------------------------------------------------------------
def gini(p):
    """G(p) = 1 - sum_k p_k^2."""
    return 1.0 - np.sum(p * p, axis=-1)


def entropy(p):
    """H(p) = - sum_k p_k log2 p_k (quy ước 0*log0 = 0)."""
    safe = np.where(p > 0, p, 1.0)
    return -np.sum(p * np.log2(safe), axis=-1)


_CRITERIA = {"gini": gini, "entropy": entropy}


# --------------------------------------------------------------------------
# Lớp cơ sở
# --------------------------------------------------------------------------
class _BaseTree:
    _is_classifier = False

    def __init__(self, criterion, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features=None, ccp_alpha=0.0, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    # ----- các hàm do lớp con định nghĩa ---------------------------------
    def _make_stats(self, y, w):
        raise NotImplementedError

    def _impurity(self, s):
        raise NotImplementedError

    def _weight(self, s):
        raise NotImplementedError

    def _value(self, s):
        raise NotImplementedError

    # ----- tiện ích -------------------------------------------------------
    def _resolve_max_features(self, d):
        mf = self.max_features
        if mf is None:
            k = d
        elif mf == "sqrt":
            k = int(np.sqrt(d))
        elif mf == "log2":
            k = int(np.log2(d))
        elif isinstance(mf, (int, np.integer)):
            k = int(mf)
        elif isinstance(mf, float):
            k = int(mf * d)
        else:
            raise ValueError(f"max_features không hợp lệ: {mf!r}")
        return min(d, max(1, k))

    # ----- huấn luyện -----------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều (n_mẫu, n_đặc_trưng)")
        y = np.asarray(y)
        n, d = X.shape
        if y.shape[0] != n:
            raise ValueError("X và y khác số mẫu")
        if np.isnan(X).any():
            raise ValueError("Bản cài đặt này chưa hỗ trợ giá trị NaN trong X")
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("sample_weight không hợp lệ")

        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = d
        self._max_features_ = self._resolve_max_features(d)
        S = self._make_stats(y, w)
        self._build(X, S, rng)
        if self.ccp_alpha and self.ccp_alpha > 0:
            self.prune(self.ccp_alpha, inplace=True)
        return self

    def _best_split(self, X, S, idx, node_weight, node_imp, rng):
        """Tìm (đặc trưng, ngưỡng, độ giảm tạp chất có trọng số) tốt nhất cho nút."""
        n = idx.size
        d = X.shape[1]
        k = self._max_features_
        msl = self.min_samples_leaf

        pos = np.arange(1, n)                       # vị trí cắt: i+1 mẫu bên trái
        base_valid = (pos >= msl) & (n - pos >= msl)
        if not base_valid.any():
            return None

        Sn = S[idx]
        total = Sn.sum(axis=0)
        perm = rng.permutation(d)
        ptr, n_visited = 0, 0
        best_child, best = np.inf, None

        # Duyệt đặc trưng theo từng lô; đặc trưng hằng số không tính vào k
        # (giống scikit-learn): tiếp tục lấy thêm đặc trưng cho đến khi đủ k
        # đặc trưng không hằng số, hoặc đã có phép tách hợp lệ.
        while ptr < d and (n_visited < k or best is None):
            need = max(k - n_visited, 1)
            batch = perm[ptr:ptr + need]
            ptr += need

            Xb = X[np.ix_(idx, batch)]                         # (n, b)
            order = np.argsort(Xb, axis=0, kind="stable")      # (n, b)
            xs = np.take_along_axis(Xb, order, axis=0)
            distinct = xs[1:] > xs[:-1]                        # (n-1, b)
            n_visited += int(distinct.any(axis=0).sum())
            valid = distinct & base_valid[:, None]
            if not valid.any():
                continue

            left = np.cumsum(Sn[order], axis=0)[:-1]           # (n-1, b, m)
            right = total - left
            with np.errstate(divide="ignore", invalid="ignore"):
                child = (self._weight(left) * self._impurity(left)
                         + self._weight(right) * self._impurity(right))
            child = np.where(valid, child, np.inf)
            flat = int(np.argmin(child))
            i, j = divmod(flat, child.shape[1])
            if child[i, j] < best_child:
                best_child = child[i, j]
                lo, hi = xs[i, j], xs[i + 1, j]
                t = 0.5 * (lo + hi)
                if not (lo <= t < hi):                         # phòng sai số số học
                    t = lo
                best = (int(batch[j]), float(t))

        if best is None:
            return None
        gain = node_weight * node_imp - best_child
        return best[0], best[1], gain

    def _build(self, X, S, rng):
        n = X.shape[0]
        feat, thr, left, right = [], [], [], []
        imp, wt, ns, val = [], [], [], []

        def add(idx):
            s = S[idx].sum(axis=0)
            feat.append(LEAF)
            thr.append(np.nan)
            left.append(LEAF)
            right.append(LEAF)
            imp.append(float(self._impurity(s)))
            wt.append(float(self._weight(s)))
            ns.append(idx.size)
            val.append(self._value(s))
            return len(feat) - 1

        all_idx = np.arange(n)
        root = add(all_idx)
        w_total = wt[0]
        max_depth = np.inf if self.max_depth is None else self.max_depth
        min_split = max(self.min_samples_split, 2 * self.min_samples_leaf)

        stack = [(root, all_idx, 0)]
        while stack:
            node, idx, depth = stack.pop()
            if depth >= max_depth or idx.size < min_split or imp[node] <= _EPS * imp[0]:
                continue
            res = self._best_split(X, S, idx, wt[node], imp[node], rng)
            if res is None:
                continue
            f, t, gain = res
            if gain / w_total < self.min_impurity_decrease - 1e-12:
                continue
            mask = X[idx, f] <= t
            li, ri = idx[mask], idx[~mask]
            l, r = add(li), add(ri)
            feat[node], thr[node], left[node], right[node] = f, t, l, r
            stack.append((l, li, depth + 1))
            stack.append((r, ri, depth + 1))

        self.feature_ = np.array(feat, dtype=np.int64)
        self.threshold_ = np.array(thr, dtype=np.float64)
        self.left_ = np.array(left, dtype=np.int64)
        self.right_ = np.array(right, dtype=np.int64)
        self.impurity_ = np.array(imp, dtype=np.float64)
        self.weight_ = np.array(wt, dtype=np.float64)
        self.n_samples_ = np.array(ns, dtype=np.int64)
        self.value_ = np.array(val, dtype=np.float64)

    # ----- dự đoán --------------------------------------------------------
    def apply(self, X):
        """Trả về chỉ số nút lá mà mỗi mẫu rơi vào."""
        X = np.asarray(X, dtype=np.float64)
        node = np.zeros(X.shape[0], dtype=np.int64)
        while True:
            act = np.flatnonzero(self.left_[node] != LEAF)
            if act.size == 0:
                return node
            nd = node[act]
            go_left = X[act, self.feature_[nd]] <= self.threshold_[nd]
            node[act] = np.where(go_left, self.left_[nd], self.right_[nd])

    # ----- thuộc tính cây -------------------------------------------------
    @property
    def node_count(self):
        return int(self.left_.size)

    @property
    def n_leaves(self):
        return int((self.left_ == LEAF).sum())

    def get_depth(self):
        depth = np.zeros(self.node_count, dtype=np.int64)
        for t in range(self.node_count):          # con có id > cha
            if self.left_[t] != LEAF:
                depth[self.left_[t]] = depth[self.right_[t]] = depth[t] + 1
        return int(depth.max())

    @property
    def feature_importances_(self):
        """Độ giảm tạp chất có trọng số, cộng theo đặc trưng, chuẩn hóa tổng = 1."""
        out = np.zeros(self.n_features_in_)
        inner = np.flatnonzero(self.left_ != LEAF)
        l, r = self.left_[inner], self.right_[inner]
        dec = (self.weight_[inner] * self.impurity_[inner]
               - self.weight_[l] * self.impurity_[l]
               - self.weight_[r] * self.impurity_[r])
        np.add.at(out, self.feature_[inner], dec)
        s = out.sum()
        return out / s if s > 0 else out

    # ----- cắt tỉa theo độ phức tạp chi phí -------------------------------
    def _weakest_link(self, max_alpha=np.inf):
        """
        Cắt tỉa "mắt xích yếu nhất" lặp đi lặp lại.

        Với nút trong t:   g(t) = (R(t) - R(T_t)) / (|lá(T_t)| - 1)
        trong đó R(t) = (W_t / W) * impurity(t) là nguy cơ nếu t là lá,
        R(T_t) là tổng nguy cơ các lá của cây con T_t. Mỗi bước cắt nút có g nhỏ
        nhất; chỉ ancestors của nút bị cắt phải cập nhật g (dùng heap + phiên bản
        để loại các mục lỗi thời) nên tổng chi phí xấp xỉ O(#nút * độ sâu * log).
        """
        n_nodes = self.node_count
        left, right = self.left_.copy(), self.right_.copy()
        R = self.weight_ * self.impurity_ / self.weight_[0]

        parent = np.full(n_nodes, -1, dtype=np.int64)
        inner = np.flatnonzero(left != LEAF)
        parent[left[inner]] = inner
        parent[right[inner]] = inner

        n_leaves = np.ones(n_nodes, dtype=np.int64)
        sub_risk = R.copy()
        for t in range(n_nodes - 1, -1, -1):
            if left[t] != LEAF:
                n_leaves[t] = n_leaves[left[t]] + n_leaves[right[t]]
                sub_risk[t] = sub_risk[left[t]] + sub_risk[right[t]]

        def g(t):
            return (R[t] - sub_risk[t]) / (n_leaves[t] - 1)

        alive = np.ones(n_nodes, dtype=bool)
        version = np.zeros(n_nodes, dtype=np.int64)
        heap = [(g(t), int(t), 0) for t in inner]
        heapq.heapify(heap)

        alphas, impurities = [0.0], [float(sub_risk[0])]
        prev = 0.0
        while left[0] != LEAF and heap:
            gt, t, ver = heapq.heappop(heap)
            if not alive[t] or ver != version[t] or left[t] == LEAF:
                continue
            eff = max(gt, prev)                    # alpha hiệu dụng không giảm
            if eff > max_alpha:
                break
            prev = eff

            d_leaves = n_leaves[t] - 1
            d_risk = R[t] - sub_risk[t]
            stack = [left[t], right[t]]            # đánh dấu hậu duệ đã bị cắt
            while stack:
                u = stack.pop()
                alive[u] = False
                if left[u] != LEAF:
                    stack.extend((left[u], right[u]))
            left[t] = right[t] = LEAF
            n_leaves[t], sub_risk[t] = 1, R[t]

            a = parent[t]
            while a != -1:
                n_leaves[a] -= d_leaves
                sub_risk[a] += d_risk
                version[a] += 1
                heapq.heappush(heap, (g(a), int(a), int(version[a])))
                a = parent[a]

            alphas.append(eff)
            impurities.append(float(sub_risk[0]))
        return alphas, impurities, left, right

    def cost_complexity_pruning_path(self):
        """
        Dãy alpha hiệu dụng tăng dần và tổng độ tạp chất ở các lá tương ứng
        (R(T_alpha)). Cùng định dạng với scikit-learn: (ccp_alphas, impurities).
        """
        alphas, imps, _, _ = self._weakest_link()
        out_a, out_i = [], []
        for a, i in zip(alphas, imps):
            if out_a and a == out_a[-1]:
                out_i[-1] = i
            else:
                out_a.append(a)
                out_i.append(i)
        return np.array(out_a), np.array(out_i)

    def prune(self, alpha, inplace=False):
        """Cắt tỉa tới khi mọi mắt xích yếu có g(t) > alpha. Trả về cây đã cắt."""
        tree = self if inplace else copy.deepcopy(self)
        _, _, left, right = tree._weakest_link(max_alpha=alpha)
        tree._compact(left, right)
        return tree

    def _compact(self, left, right):
        """Loại các nút không còn nối với gốc, đánh số lại (cha < con)."""
        order, stack = [], [0]
        while stack:
            u = stack.pop()
            order.append(u)
            if left[u] != LEAF:
                stack.append(right[u])
                stack.append(left[u])
        order = np.array(order, dtype=np.int64)
        remap = np.full(left.size, -1, dtype=np.int64)
        remap[order] = np.arange(order.size)
        l, r = left[order], right[order]
        self.left_ = np.where(l != LEAF, remap[l], LEAF)
        self.right_ = np.where(r != LEAF, remap[r], LEAF)
        self.feature_ = self.feature_[order]
        self.threshold_ = self.threshold_[order]
        self.impurity_ = self.impurity_[order]
        self.weight_ = self.weight_[order]
        self.n_samples_ = self.n_samples_[order]
        self.value_ = self.value_[order]
        leaf = self.left_ == LEAF
        self.feature_[leaf] = LEAF
        self.threshold_[leaf] = np.nan

    # ----- in cây ra văn bản (tiện cho bài tập vẽ tay / kiểm tra) ----------
    def export_text(self, feature_names=None, decimals=3):
        names = feature_names or [f"x{j}" for j in range(self.n_features_in_)]
        lines = []

        def rec(node, depth):
            pad = "|   " * depth
            if self.left_[node] == LEAF:
                lines.append(f"{pad}|--- {self._leaf_text(node, decimals)}  "
                             f"(n={self.n_samples_[node]}, impurity={self.impurity_[node]:.{decimals}f})")
                return
            nm, t = names[self.feature_[node]], self.threshold_[node]
            lines.append(f"{pad}|--- {nm} <= {t:.{decimals}f}")
            rec(self.left_[node], depth + 1)
            lines.append(f"{pad}|--- {nm} >  {t:.{decimals}f}")
            rec(self.right_[node], depth + 1)

        rec(0, 0)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Phân loại
# --------------------------------------------------------------------------
class DecisionTreeClassifier(_BaseTree):
    _is_classifier = True

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features=None, ccp_alpha=0.0, random_state=None):
        if criterion not in _CRITERIA:
            raise ValueError("criterion phải là 'gini' hoặc 'entropy'")
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf,
                         min_impurity_decrease, max_features, ccp_alpha, random_state)

    def _make_stats(self, y, w):
        self.classes_, yi = np.unique(y, return_inverse=True)
        S = np.zeros((y.shape[0], self.classes_.size))
        S[np.arange(y.shape[0]), yi] = w
        return S

    def _impurity(self, s):
        tot = s.sum(axis=-1, keepdims=True)
        p = s / np.where(tot > 0, tot, 1.0)
        return _CRITERIA[self.criterion](p)

    def _weight(self, s):
        return s.sum(axis=-1)

    def _value(self, s):
        return s / s.sum()

    def predict_proba(self, X):
        return self.value_[self.apply(X)]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _leaf_text(self, node, decimals):
        return f"class: {self.classes_[np.argmax(self.value_[node])]}"


# --------------------------------------------------------------------------
# Hồi quy
# --------------------------------------------------------------------------
class DecisionTreeRegressor(_BaseTree):
    def __init__(self, criterion="squared_error", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features=None, ccp_alpha=0.0, random_state=None):
        if criterion not in ("squared_error", "mse"):
            raise ValueError("criterion phải là 'squared_error'")
        super().__init__("squared_error", max_depth, min_samples_split, min_samples_leaf,
                         min_impurity_decrease, max_features, ccp_alpha, random_state)

    def _make_stats(self, y, w):
        y = y.astype(np.float64)
        # Trừ trung bình để cumsum ổn định số học; cộng lại ở _value.
        self._offset = float(np.average(y, weights=w))
        yc = y - self._offset
        return np.column_stack([w, w * yc, w * yc * yc])

    def _impurity(self, s):
        s0 = s[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s[..., 1] / s0
            var = s[..., 2] / s0 - mean * mean
        return np.where(s0 > 0, np.maximum(var, 0.0), 0.0)

    def _weight(self, s):
        return s[..., 0]

    def _value(self, s):
        return np.array([s[1] / s[0] + self._offset])

    def predict(self, X):
        return self.value_[self.apply(X), 0]

    def score(self, X, y):
        """Hệ số xác định R^2."""
        y = np.asarray(y, dtype=np.float64)
        resid = np.sum((y - self.predict(X)) ** 2)
        return float(1.0 - resid / np.sum((y - y.mean()) ** 2))

    def _leaf_text(self, node, decimals):
        return f"value: {self.value_[node, 0]:.{decimals}f}"
