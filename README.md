# Hybrid Movie Recommendation System 🎬

This project presents a Hybrid Movie Recommendation System developed using Machine Learning techniques to provide personalized and intelligent movie recommendations. The system combines both Content-Based Filtering and Collaborative Filtering approaches to improve recommendation accuracy and overall user experience.

The Content-Based Filtering model analyzes movie information such as titles and genres using TF-IDF vectorization and Cosine Similarity to identify movies with similar characteristics. In addition, Collaborative Filtering is implemented using Singular Value Decomposition (SVD), which learns user preferences from movie ratings and predicts unseen movies based on user behavior patterns.

To achieve better recommendation performance, a Hybrid Recommendation Engine was developed by combining the outputs of both models using weighted scoring. This approach helps generate more reliable and personalized recommendations for users.

The project also includes an interactive web application built with Streamlit, allowing users to:

* Search for movies
* Select different user profiles
* Generate hybrid movie recommendations
* Control the number of recommended movies
* View recommendation scores and evaluation metrics

The system was evaluated using multiple performance metrics, including:

* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)
* Precision
* Recall
* F1-Score

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

## Machine Learning Techniques

* TF-IDF Vectorization
* Cosine Similarity
* Singular Value Decomposition (SVD)
* Hybrid Recommendation Systems

This project demonstrates the practical implementation of recommendation systems and highlights the effectiveness of combining multiple machine learning techniques to create accurate and personalized intelligent systems.
