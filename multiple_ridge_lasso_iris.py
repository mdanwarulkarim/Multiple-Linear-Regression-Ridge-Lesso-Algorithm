import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score)

# 1. Data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
X = df.drop(columns="sepal length (cm)")
y = df["sepal length (cm)"]
print(X.corr().round(2))

# 2. Split, then scale (fit on train only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. Three models on the same scaled data
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=10),
    "Lasso": Lasso(alpha=0.05),
}
rows, coefs = [], {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    coefs[name] = model.coef_
    rows.append({
        "Model": name,
        "MSE": mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred)})
print(pd.DataFrame(rows).round(3))
print(pd.DataFrame(coefs, index=X.columns).round(3))

# 4. Shuffled 5-fold CV (fairer than one split)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), model)
    score = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    print(name, "CV R2:", score.mean().round(3))
