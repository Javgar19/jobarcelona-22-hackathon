import numpy
import pandas
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train (1).csv")
df_test = pd.read_csv("test_x.csv")
df["Time"] = df["Hour"] * 60 + df["Minutes"]


# Preprocessing
features = ['Time', 'Sensor_beta', 'Sensor_gamma', 'Sensor_alpha_plus']
target = 'Insect'

X = df[features]
y = df[target]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# Model creation and fit
final_model = tf.keras.Sequential()
final_model.add(tf.keras.layers.Dense(32, input_dim=4, activation='relu'))
final_model.add(tf.keras.layers.Dense(32, activation='relu'))
final_model.add(tf.keras.layers.Dropout(0.2))
final_model.add(tf.keras.layers.Dense(32, activation='relu'))
final_model.add(tf.keras.layers.Dense(32, activation='relu'))
final_model.add(tf.keras.layers.Dropout(0.2))
final_model.add(tf.keras.layers.Dense(3, activation='softmax'))

final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y = tf.keras.utils.to_categorical(y)

final_model.fit(X_scaled, y, verbose=2, epochs=50, batch_size=16)


# Preprocessing
df_test["Time"] = df_test["Hour"] * 60 + df_test["Minutes"]

features = ['Time', 'Sensor_beta', 'Sensor_gamma', 'Sensor_alpha_plus']

X_final_test = df[features]

scaler = StandardScaler()
scaler.fit(X_final_test)
X_scaled_final_test = scaler.transform(X_final_test)

# Prediction and save the results
results = np.argmax(final_model.predict(X_scaled_final_test), axis=1)
results_df = pd.DataFrame(data=results, columns=['Insect'])
results_df.head()
results_df.to_csv("results.csv")
