import pandas as pd

colnames = ['id', 'lat', 'long', 'text']

training_set_df = pd.read_csv("pml-2020-unibuc/training.txt", names=colnames)
test_set_df = pd.read_csv("pml-2020-unibuc/test.txt")
validation_set_df = pd.read_csv("pml-2020-unibuc/validation.txt",  names=colnames)

train_text_series = training_set_df.text
train_lat_series = training_set_df.lat
train_long_series = training_set_df.long

val_text_series = validation_set_df.text

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(lowercase = 'true')
matrix_text_train = tfidf_vectorizer.fit_transform(train_text_series)
matrix_text_val = tfidf_vectorizer.fit_transform(val_text_series)

train_text_array = matrix_text_train.toarray()
val_text_toarray = matrix_text_val.toarray()
train_text_array
train_lat_array = train_lat_series.to_numpy()
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

X = train_text_array
y_lat = train_lat_array

reg.fit(X, y_lat)

