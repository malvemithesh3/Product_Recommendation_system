# Product_Recommendation_system


## Objective

This project demonstrates a simple recommendation system to suggest relevant products to users based on their past behavior. The model simulates how machine learning can enhance retail experiences by offering personalized recommendations.

### Dataset

Source: Retail dataset (Retail.csv)

### Fields:

1. User ID: Unique identifier for each user

2. Product ID: Unique identifier for each product

3. Product Category: Category of the product

4. Purchase Date: Date of the transaction

5. Rating: User’s rating of the product

## Approach

### Data Preprocessing:

* Converted Purchase Date to a datetime format and handled missing values by removing rows with null values.

* Created a user-item matrix to store user ratings for products, which is essential for collaborative filtering.

### Model Development:

* Used item-based collaborative filtering to calculate the similarity between products.

* Calculated cosine similarity between product vectors to identify similar products that could be recommended to a user.

* Created a recommendation function that returns the most similar products for a given product ID.


## Evaluation:

* We split the data into training and testing sets for validation.

* You can further evaluate this model with metrics such as RMSE on predicted ratings.


### Challenges

1. Ensuring all Product IDs had consistent data types.

2. This model may struggle with users or products that have very few ratings.

### How to Run

Install necessary libraries:

pip install pandas scikit-learn numpy

Run the Python script to see product recommendations based on collaborative filtering.

### Building an API or Dashboard

To implement a simple API for product recommendations, consider using FastAPI.

Install Required Packages

For FastAPI:

pip install fastapi uvicorn

Run app.py from Command Prompt

Open Command Prompt:

Navigate to the Directory:

Run the Server:

uvicorn app:app --reload
