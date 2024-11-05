import pandas as pd

df = pd.read_csv("retail.csv")
df

df.info()
df.describe()

user_purchase_counts = df.groupby('User ID').size()
user_purchase_counts.describe()
product_popularity = df.groupby('Product ID').size()
product_popularity.describe()
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
df = pd.get_dummies(df, columns=['Product Category'], drop_first=True)
df

df.fillna(method='ffill', inplace=True)
df

user_item_matrix = df.pivot_table(index='User ID', columns='Product ID', values='Rating').fillna(0)
user_item_matrix

from sklearn.metrics.pairwise import cosine_similarity

product_similarity = cosine_similarity(user_item_matrix.T)
product_similarity_df = pd.DataFrame(product_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
product_similarity_df

user_item_matrix.columns = user_item_matrix.columns.astype(int)
product_similarity_df.index = product_similarity_df.index.astype(int)
product_similarity_df.columns = product_similarity_df.columns.astype(int)

def recommend_products(product_id, num_recommendations):
    if product_id not in product_similarity_df.index:
        print("Product not found in the dataset.")
        return []
    similar_products = product_similarity_df[product_id].sort_values(ascending=False)[1:num_recommendations+1]
    return similar_products.index.tolist()


recommend_products(101, num_recommendations=5)

train_data = user_item_matrix.sample(frac=0.8, random_state=42)
test_data = user_item_matrix.drop(train_data.index)

def rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

from sklearn.metrics import mean_squared_error
from math import sqrt

pred_ratings = cosine_similarity(train_data.T)
rmse(pred_ratings, test_data.values)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Product Recommendation API"}

@app.get("/recommend/{product_id}")
def recommend(product_id: int, num_recommendations: int = 5):
    recommendations = recommend_products(product_id, num_recommendations)
    return {"recommended_products": recommendations}








