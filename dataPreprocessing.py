import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# 1-1
fires = pd.read_csv("sanbul2district-divby100.csv", sep=",")
fires['burned_area'] = np.log(fires['burned_area'] + 1)

#1-2
print(fires.head())
print(fires.info())
print(fires.describe())
print(fires['month'].value_counts())
print(fires['day'].value_counts())

#1-3
fires.hist(bins=50, figsize=(15,10))
plt.show()

#1-4
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fires_raw = pd.read_csv("sanbul2district-divby100.csv")
axs[0].hist(fires_raw["burned_area"], bins=50)
axs[0].set_title("Original burned_area")

axs[1].hist(fires["burned_area"], bins=50)
axs[1].set_title("Log Transformed burned_area")
plt.annotate("ln(burned_area+1)", xy=(5, 150), xytext=(2.5, 200),
             arrowprops=dict(facecolor='blue', shrink=0.05),
             fontsize=12, color='blue', ha='center')

plt.tight_layout()
plt.show()

#1-5
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

test_set.head()
fires["month"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

print("\nMonth category proportion: \n",
      strat_test_set["month"].value_counts() / len(strat_test_set))

print("\nOverall month category proportion: \n",
      fires["month"].value_counts() / len(fires))

#1-6
attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]
scatter_matrix(fires[attributes], figsize=(12,8))
plt.show()

#1-7
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
           s=fires["max_temp"], label="max_temp",
           c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

#1-8
fires = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()
fires_num = fires.drop(["month", "day"], axis=1) 

fires_month = strat_train_set[["month"]]
fires_day = strat_train_set[["day"]]

cat_month_encoder = OneHotEncoder()
cat_day_encoder = OneHotEncoder()

fires_month_1hot = cat_month_encoder.fit_transform(fires_month)
fires_day_1hot = cat_day_encoder.fit_transform(fires_day)

print("\n=== 1-8 OneHotEncoder results ===")
print("cat_month_encoder.categories_:")
print(fires_month_1hot.toarray())
print(cat_month_encoder.categories_)

print("cat_day_encoder.categories_:")
print(fires_day_1hot.toarray())
print(cat_day_encoder.categories_)

#1-9
print("\n\n#########################################################")
print("Now let's build a pipline for preprocessing the numerical attributes:")
num_attribs = list(fires_num)
cat_attribs = ["month", "day"] 

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

fires_prepared = full_pipeline.fit_transform(fires)

test_fires = strat_test_set.drop(["burned_area"], axis=1)
fires_test_prepared = full_pipeline.transform(test_fires)
fires_test_labels = strat_test_set["burned_area"].copy()
print(fires_prepared.shape)
print(fires_prepared[:5])


#2
X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))

model.save("fires_model.keras")

X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2) : \n", 
      np.round(model.predict(X_new), 2))