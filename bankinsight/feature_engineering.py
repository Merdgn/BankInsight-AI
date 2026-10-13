
from sklearn.base import BaseEstimator, TransformerMixin

class BankFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["has_previous_contact"] = (
            X["pdays"] >= 0
        ).astype(int)

        X["pdays"] = X["pdays"].replace(-1, 0)

        return X
