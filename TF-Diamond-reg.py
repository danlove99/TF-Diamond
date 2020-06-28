import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd 
from sklearn.model_selection import train_test_split


# Function to preprocess dataset
def preprocess(dataframe):
	dataframe.drop(['x', 'y', 'z'], axis=1)
	dataframe = pd.get_dummies(dataframe, columns=['cut', 'color', 'clarity'])
	dataframe.pop('Unnamed: 0')
	dataframe = dataframe.rename(columns={'cut_Very Good' : 'cut_Very_Good'})
	return dataframe

# Function to create tensorflow dataset from pandas dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=16):
  labels = dataframe.pop('price')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

# Load in data and pass to preprocessing function
df = preprocess(pd.read_csv('diamonds.csv'))

# Split into train and test
train=df.sample(frac=0.8,random_state=200) 
test=df.drop(train.index)
predict_df = df.tail(1)
correct_prediction = predict_df.pop('price')
df.drop(df.tail(1).index,inplace=True)
# Make compatible with TensorFlow
train_ds = df_to_dataset(train)
test_ds = df_to_dataset(test, shuffle=False)

#predict_ds = tf.data.Dataset.from_tensor_slices((dict(dataframe))

# Create feature layer
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

feature_layer = tf.keras.layers.DenseFeatures(my_feature_columns)


# Create and comopile model
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='MSE',
              metrics=['MSE'])

# Train
model.fit(train_ds,
          epochs=2)

# Evaluate with test dataset          
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

prediction = model.predict(dict(predict_df))
print(prediction * 100)
print(correct_prediction)