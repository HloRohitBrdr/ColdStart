import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os


users_df = None
products_df = None
ratings_df = None
model = None
index = None
user_features = None


def initialize_recommendation_system():
    """Initialize the recommendation system components lazily"""
    global users_df, products_df, ratings_df, model, index, user_features

    if users_df is not None:
        return  # Already initialized

    print("🚀 Loading recommendation system...")

    # Load data
    users_df = pd.read_csv("data/Updated_Users_Dataset_with_Demographics.csv")
    products_df = pd.read_csv("data/products_large.csv")
    ratings_df = pd.read_csv("data/ratings_large.csv")

    print(f" Data loaded: Users {len(users_df)}, Products {len(products_df)}, Ratings {len(ratings_df)}")

    # Embedding model
    print(" Loading AI model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create FAISS index
    print(" Building search index...")
    descriptions = products_df['name'].fillna("").tolist()
    embeddings = model.encode(descriptions, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    # Preprocess user data
    print("👥 Processing user features...")
    gender_encoded = pd.get_dummies(users_df['gender'], prefix='gender')
    location_map = {loc: idx for idx, loc in enumerate(users_df['location'].unique())}
    users_df['location_encoded'] = users_df['location'].map(location_map)
    scaler = MinMaxScaler()
    users_df['age_scaled'] = scaler.fit_transform(users_df[['age']])
    interests_split = users_df['interests'].str.get_dummies(sep=',')

    user_features = pd.concat([
        users_df[['user_id', 'location_encoded', 'age_scaled']],
        gender_encoded,
        interests_split
    ], axis=1)

    print(" Recommendation system ready!")


def recommend_for_user(user_id, query, top_n=5):
    initialize_recommendation_system()  # Ensure system is loaded

    if user_id not in users_df['user_id'].values:
        return semantic_search(query)

    user_vec = user_features[user_features['user_id'] == user_id].drop(columns=['user_id']).values
    all_vecs = user_features.drop(columns=['user_id']).values

    sim = cosine_similarity(user_vec, all_vecs)[0]
    user_features['similarity'] = sim
    similar_users = user_features.sort_values(by='similarity', ascending=False).head(10)['user_id'].values
    sim_ratings = ratings_df[ratings_df['user_id'].isin(similar_users)]

    merged = sim_ratings.merge(user_features[['user_id', 'similarity']], on='user_id')
    merged['weighted_rating'] = merged['rating'] * merged['similarity']
    agg = merged.groupby('product_id')['weighted_rating'].sum().reset_index().sort_values(by='weighted_rating',
                                                                                          ascending=False)

    results = products_df[products_df['product_id'].isin(agg.head(top_n)['product_id'])]
    return results[['name', 'category', 'price']].to_dict(orient="records")


def semantic_search(query, top_k=5):
    initialize_recommendation_system()  # Ensure system is loaded

    query_vec = model.encode([query])[0].astype("float32")
    _, indices = index.search(np.array([query_vec]), top_k)
    results = products_df.iloc[indices[0]]
    return results[['name', 'category', 'price']].to_dict(orient="records")
