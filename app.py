import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score


# Load data
movies = pd.read_csv("movies.csv").dropna()
ratings = pd.read_csv("ratings.csv").dropna()

movies["genres"] = movies["genres"].fillna("")


# Content-Based Filtering
movies["features"] = movies["title"] + " " + movies["genres"]

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["features"])

similarity = cosine_similarity(tfidf_matrix)
movie_index = pd.Series(movies.index, index=movies["title"]).drop_duplicates()


def content_recommend(movie_name, n=10):
    idx = movie_index[movie_name]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n + 1]

    movie_ids = [x[0] for x in scores]
    movie_scores = [x[1] for x in scores]

    result = movies.iloc[movie_ids][["movieId", "title", "genres"]].copy()
    result["content_score"] = movie_scores

    return result


# Collaborative Filtering using SVD
user_movie = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

svd = TruncatedSVD(n_components=20, random_state=42)

user_features = svd.fit_transform(user_movie)
movie_features = svd.components_

predicted_ratings = np.dot(user_features, movie_features)

predicted_df = pd.DataFrame(
    predicted_ratings,
    index=user_movie.index,
    columns=user_movie.columns
)


def user_recommend(user_id, n=10):
    user_scores = predicted_df.loc[user_id]

    watched = ratings[ratings["userId"] == user_id]["movieId"]
    user_scores = user_scores.drop(watched, errors="ignore")

    top_movies = user_scores.sort_values(ascending=False).head(n)

    result = movies[movies["movieId"].isin(top_movies.index)][
        ["movieId", "title", "genres"]
    ].copy()

    result["svd_score"] = result["movieId"].map(top_movies)

    return result.sort_values("svd_score", ascending=False)


# Hybrid Recommendation
def hybrid_recommend(user_id, movie_name, n=10):
    content_movies = content_recommend(movie_name, 30)
    user_movies = user_recommend(user_id, 30)

    final = pd.merge(
        content_movies,
        user_movies,
        on=["movieId", "title", "genres"],
        how="outer"
    ).fillna(0)

    if final["content_score"].max() != 0:
        final["content_score"] = final["content_score"] / final["content_score"].max()

    if final["svd_score"].max() != 0:
        final["svd_score"] = final["svd_score"] / final["svd_score"].max()

    final["hybrid_score"] = (
        0.5 * final["content_score"] +
        0.5 * final["svd_score"]
    )

    return final.sort_values("hybrid_score", ascending=False).head(n)


# Evaluation
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

train_matrix = train.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

eval_svd = TruncatedSVD(n_components=20, random_state=42)

eval_user_features = eval_svd.fit_transform(train_matrix)
eval_movie_features = eval_svd.components_

eval_predictions = np.dot(eval_user_features, eval_movie_features)

eval_pred_df = pd.DataFrame(
    eval_predictions,
    index=train_matrix.index,
    columns=train_matrix.columns
)

global_avg = train["rating"].mean()
movie_avg = train.groupby("movieId")["rating"].mean()
user_avg = train.groupby("userId")["rating"].mean()

y_true = []
y_pred = []

for _, row in test.iterrows():
    user_id_test = row["userId"]
    movie_id_test = row["movieId"]

    y_true.append(row["rating"])

    if user_id_test in eval_pred_df.index and movie_id_test in eval_pred_df.columns:
        pred = eval_pred_df.loc[user_id_test, movie_id_test]
    elif movie_id_test in movie_avg:
        pred = movie_avg[movie_id_test]
    elif user_id_test in user_avg:
        pred = user_avg[user_id_test]
    else:
        pred = global_avg

    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Convert ratings to liked / not liked
actual_liked = (y_true >= 4).astype(int)

# Dynamic threshold so classification metrics do not become all zeros
threshold = np.percentile(y_pred, 60)
predicted_liked = (y_pred >= threshold).astype(int)

precision = precision_score(actual_liked, predicted_liked, zero_division=0)
recall = recall_score(actual_liked, predicted_liked, zero_division=0)
f1 = f1_score(actual_liked, predicted_liked, zero_division=0)


# Streamlit App
st.title("Hybrid Movie Recommendation System 🎬")

st.write("This app recommends movies using Content-Based Filtering and SVD.")

user_id = st.selectbox("Choose User ID", sorted(ratings["userId"].unique()))

# Search feature
search_text = st.text_input("Search for a movie")

if search_text:
    movie_list = movies[
        movies["title"].str.contains(search_text, case=False, na=False)
    ]["title"].tolist()
else:
    movie_list = sorted(movies["title"].unique())

if movie_list:
    movie_name = st.selectbox("Choose Movie", movie_list)
else:
    st.warning("No movies found")
    movie_name = None

num = st.slider("Number of movies", 5, 15, 10)

if st.button("Recommend"):
    if movie_name:
        st.subheader("Hybrid Recommendations")

        result = hybrid_recommend(user_id, movie_name, num)

        st.dataframe(
            result[["title", "genres", "content_score", "svd_score", "hybrid_score"]],
            width="stretch"
        )
    else:
        st.error("Please choose a valid movie first")


st.subheader("Evaluation")

st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1-Score: {f1:.2f}")