"""Exact greedy Newton tree boosting for squared-error regression."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass
class Node:
    weight: float
    feature: int = -1
    threshold: float = 0.0
    gain: float = 0.0
    left: Node | None = None
    right: Node | None = None


class XGBRegressorCustom:
    """Fit finite numeric features using gradient and Hessian sums."""

    def __init__(self, n_estimators=30, max_depth=3, learning_rate=0.1,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
        if (not isinstance(n_estimators, Integral) or n_estimators < 1
                or not isinstance(max_depth, Integral) or max_depth < 0
                or not 0 < learning_rate <= 1):
            raise ValueError("Invalid number of trees, depth or learning rate")
        regularization = np.array([reg_lambda, gamma, min_child_weight])
        if not np.isfinite(regularization).all() or (regularization < 0).any():
            raise ValueError("Regularization and minimum child weight must be nonnegative")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def _grow(self, X, gradient, hessian, rows, depth):
        g, h = gradient[rows].sum(), hessian[rows].sum()
        penalty = self.reg_lambda
        node = Node(float(-g / (h + penalty)))
        if depth == self.max_depth or len(rows) < 2:
            return node
        parent_score = g**2 / (h + penalty)
        best_rows = None
        for feature in range(X.shape[1]):
            ordered = rows[np.argsort(X[rows, feature], kind="stable")]
            values = X[ordered, feature]
            gl = np.cumsum(gradient[ordered])[:-1]
            hl = np.cumsum(hessian[ordered])[:-1]
            gr, hr = g - gl, h - hl
            valid = ((values[:-1] < values[1:]) & (hl >= self.min_child_weight)
                     & (hr >= self.min_child_weight))
            gains = 0.5 * (gl**2 / (hl + penalty) + gr**2 / (hr + penalty)
                           - parent_score) - self.gamma
            gains[~valid] = -np.inf
            cut = int(np.argmax(gains))
            if gains[cut] > node.gain:
                lower, upper = values[cut:cut + 2]
                threshold = lower / 2 + upper / 2
                node.feature = feature
                node.threshold = float(threshold if threshold < upper else lower)
                node.gain = float(gains[cut])
                best_rows = ordered[:cut + 1], ordered[cut + 1:]
        if best_rows is not None:
            node.left = self._grow(X, gradient, hessian, best_rows[0], depth + 1)
            node.right = self._grow(X, gradient, hessian, best_rows[1], depth + 1)
        return node

    @staticmethod
    def _predict_tree(tree, X):
        result = np.empty(len(X))

        def visit(node, rows):
            if node.feature < 0:
                result[rows] = node.weight
                return
            left = X[rows, node.feature] <= node.threshold
            visit(node.left, rows[left])
            visit(node.right, rows[~left])

        visit(tree, np.arange(len(X)))
        return result

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if (X.ndim != 2 or not X.shape[0] or not X.shape[1] or y.shape != (len(X),)
                or not np.isfinite(X).all() or not np.isfinite(y).all()):
            raise ValueError("Expected aligned, finite, nonempty numeric X and y")
        self.n_features_in_ = X.shape[1]
        self.base_score_ = float(y.mean())
        self.trees_ = []
        prediction = np.full(len(X), self.base_score_)
        hessian = np.ones(len(X))
        rows = np.arange(len(X))
        for _ in range(self.n_estimators):
            tree = self._grow(X, prediction - y, hessian, rows, 0)
            self.trees_.append(tree)
            prediction += self.learning_rate * self._predict_tree(tree, X)
        return self

    def predict(self, X):
        if not hasattr(self, "trees_"):
            raise ValueError("Fit the model before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Expected finite X with the fitted feature count")
        prediction = np.full(len(X), self.base_score_)
        for tree in self.trees_:
            prediction += self.learning_rate * self._predict_tree(tree, X)
        return prediction
