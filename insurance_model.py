import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

insurance=pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
insurance

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

#colum transformer

ct=make_column_transformer(
    (MinMaxScaler(),["age","bmi","children"]),
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
)

X=insurance.drop("charges",axis=1)
y=insurance["charges"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

ct.fit(X_train)

X_train_processed=ct.transform(X_train)
X_test_processed=ct.transform(X_test)

X_train_processed.shape

tf.random.set_seed(42)

insurance_model=tf.keras.Sequential([
     tf.keras.layers.Dense(100,activation="relu"),
     tf.keras.layers.Dense(10,activation="relu"),
     tf.keras.layers.Dense(1)
 ])

insurance_model.compile(loss="mse",
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])
insurance_model.fit(X_train_processed,y_train,epochs=200, verbose=0)

insurance_model.evaluate(X_test_processed,y_test)