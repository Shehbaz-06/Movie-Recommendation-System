#1. Always import relevant libraries to your goals
import numpy as np
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#2. Load the datasets 
links=pd.read_csv("links.csv")
movies=pd.read_csv("movies.csv")
ratings=pd.read_csv("ratings.csv")
tags=pd.read_csv("tags.csv")


#3. Inspect the datasets
print("Links Dataset Info:")
print(links.info())
print("\nMovies Dataset Info:")
print(movies.info())
print("\nRatings Dataset Info:")
print(ratings.info())
print("\nTags Dataset Info:")
print(tags.info())

# Check for missing values
print("\nMissing values:")
print("Links:\n", links.isnull().sum())
print("Movies:\n", movies.isnull().sum())
print("Ratings:\n", ratings.isnull().sum())
print("Tags:\n", tags.isnull().sum())

#4. Merging the datasets
data=pd.merge(ratings, movies,on='movieId')
# Merge tags ( for extra insight)
data = pd.merge(data, tags [['movieId', 'tag']], on='movieId', how= 'left')
# finally lets also merge links
data = pd.merge(data,links, on='movieId', how='left')
print(data.head())
print(data.tail())


#5. Create a User-Item Matrix + Linking similarity

# Create a user-item matrix

user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')


# Fill missing values with 0 (signifying user didn't rate)
user_item_matrix = user_item_matrix.fillna(0)

# Compute user similarity matrix
# cosine similarity measures how close users are based on rating patterns.
user_similarity= cosine_similarity(user_item_matrix)

# Convert similarity array to Dataframe for readability
user_similarity_df = pd.DataFrame(user_similarity, index= user_item_matrix.index, columns = user_item_matrix.index)

print("User similarity matrix created successfully!")
print(user_similarity_df.head())

def get_recommendations(user_id, user_item_matrix, user_similarity_df, k=5):
    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:6]  # top 5 similar users

    # Get all movies rated by similar users
    similar_users_ratings = user_item_matrix.loc[similar_users]

    # Compute average rating per movie among similar users
    mean_ratings = similar_users_ratings.mean(axis=0)

    # Get movies user hasn't rated yet
    user_seen = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = mean_ratings.drop(user_seen).sort_values(ascending=False).head(k)

    return recommendations

# Example: recommend for user 10
recommendations = get_recommendations(10, user_item_matrix, user_similarity_df)
print("\nRecommended Movies for User 10:")
print(recommendations)

def precision_at_k(user_id, user_item_matrix, recommendations, k=5, threshold=3.5):
    actual_ratings = user_item_matrix.loc[user_id]
    relevant = actual_ratings[actual_ratings >= threshold].index
    recommended = recommendations.index[:k]

    true_positives = len(set(recommended).intersection(set(relevant)))
    precision = true_positives / k
    return precision

precision = precision_at_k(10, user_item_matrix, recommendations)
print(f"\nPrecision@K for user 10: {precision:.2f}")

# Visualization
plt.figure(figsize=(8,5))
recommendations.plot(kind='barh', color='orange')
plt.title("Top 5 Movie Recommendations for User 10")
plt.xlabel("Predicted Rating Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
