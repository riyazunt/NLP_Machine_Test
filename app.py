import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

df_tr = pd.read_csv('new_train.csv')
df_train = df_tr[['cleanText', 'number_of_times_prescribed', 'effectiveness_rating', 'base_score']]
df_train.dropna(inplace=True)
vt = TfidfVectorizer()
dt = vt.fit_transform(df_train['cleanText'])
vt_df = pd.DataFrame(dt.todense(), columns=vt.vocabulary_.keys())
vt_df[['number_of_times_prescribed', 'effectiveness_rating']] = df_train[['number_of_times_prescribed', 'effectiveness_rating']]
y = df_train['base_score']
X_train, X_test, y_train, y_test = train_test_split(vt_df, y, test_size=0.2, random_state=101)
