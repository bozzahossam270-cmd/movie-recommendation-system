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


# Content based model
movies["features"] = movies["title"] + " " + movies["genres"]

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["features"])
similarity = cosine_similarity(tfidf_matrix)

movie_index = pd.Series(movies.index, index=movies["title"]).drop_duplicates()


def content_recommend(movie_name, n=10):
    idx = movie_index[movie_name]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n + 1]

    ids = [x[0] for x in scores]
    sim_scores = [x[1] for x in scores]

    result = movies.iloc[ids][["movieId", "title", "genres"]].copy()
    result["content_score"] = sim_scores

    return result


# Collaborative filtering using SVD
user_movie = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

svd = TruncatedSVD(n_components=20, random_state=42)
user_features = svd.fit_transform(user_movie)
movie_features = svd.components_

predicted = np.dot(user_features, movie_features)

predicted_df = pd.DataFrame(
    predicted,
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


# Hybrid recommendation
def hybrid_recommend(user_id, movie_name, n=10):
    content = content_recommend(movie_name, 30)
    collab = user_recommend(user_id, 30)

    final = pd.merge(
        content,
        collab,
        on=["movieId", "title", "genres"],
        how="outer"
    ).fillna(0)

    final["content_score"] = final["content_score"] / final["content_score"].max()
    final["svd_score"] = final["svd_score"] / final["svd_score"].max()

    final["hybrid_score"] = (
        0.5 * final["content_score"] +
        0.5 * final["svd_score"]
    )

    return final.sort_values("hybrid_score", ascending=False).head(n)


# Simple evaluation
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

y_true = test["rating"]

# simple prediction using average rating
y_pred = np.full(len(test), train["rating"].mean())

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# classification metrics
# rating 4 or more = liked movie
actual_liked = (y_true >= 4).astype(int)
predicted_liked = (y_pred >= 4).astype(int)

precision = precision_score(actual_liked, predicted_liked, zero_division=0)
recall = recall_score(actual_liked, predicted_liked, zero_division=0)
f1 = f1_score(actual_liked, predicted_liked, zero_division=0)


# Streamlit app
st.title("Hybrid Movie Recommendation System 🎬")

st.write("This app recommends movies using content based filtering and SVD.")

user_id = st.selectbox("Choose User ID", sorted(ratings["userId"].unique()))

# Search feature
search_movie = st.text_input("Search for a movie")

if search_movie:
    movie_list = movies[
        movies["title"].str.contains(search_movie, case=False, na=False)
    ]["title"].tolist()
else:
    movie_list = sorted(movies["title"].unique())

if len(movie_list) > 0:
    movie_name = st.selectbox("Choose Movie", movie_list)
else:
    st.warning("No movies found")
    movie_name = None

num = st.slider("Number of movies", 5, 15, 10)

if st.button("Recommend"):

    if movie_name is not None:
        st.subheader("Hybrid Recommendations")

        result = hybrid_recommend(user_id, movie_name, num)

        st.dataframe(
            result[["title", "genres", "content_score", "svd_score", "hybrid_score"]],
            use_container_width=True
        )
    else:
        st.error("Please choose a valid movie first")

st.subheader("Evaluation")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1-Score: {f1:.2f}")