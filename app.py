
import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #10101a;
        color: #ffffff;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #ff4b6e;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #b5b5c5;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .movie-title {
        font-size: 17px;
        font-weight: 700;
        text-align: center;
        color: white;
        margin-top: 10px;
    }

    .section-title {
        color: #ff4b6e;
        font-size: 24px;
        font-weight: bold;
        margin-top: 25px;
    }

    div.stButton > button {
        background-color: #ff4b6e;
        color: white;
        border-radius: 10px;
        border: none;
        height: 48px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #e63e60;
        color: white;
    }

    [data-testid="stSidebar"] {
        background-color: #191927;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL FILES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MOVIE_FILE = ARTIFACT_DIR / "movie_list.pkl"
SIMILARITY_FILE = ARTIFACT_DIR / "similarity.pkl"


@st.cache_resource
def load_data():

    if not MOVIE_FILE.exists():
        raise FileNotFoundError(
            f"Movie file not found: {MOVIE_FILE}"
        )

    if not SIMILARITY_FILE.exists():
        raise FileNotFoundError(
            f"Similarity file not found: {SIMILARITY_FILE}"
        )

    with open(MOVIE_FILE, "rb") as file:
        movies = pickle.load(file)

    with open(SIMILARITY_FILE, "rb") as file:
        similarity = pickle.load(file)

    if not isinstance(movies, pd.DataFrame):
        raise TypeError("movie_list.pkl must contain a pandas DataFrame.")

    if not {"title", "movie_id"}.issubset(movies.columns):
        raise ValueError(
            "Movie DataFrame must contain 'title' and 'movie_id' columns."
        )

    if len(movies) != similarity.shape[0]:
        raise ValueError(
            "Movie DataFrame and similarity matrix dimensions do not match."
        )

    return movies, similarity


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend(movie_name, movies, similarity):

    # Find movie index
    movie_indices = movies.index[
        movies["title"].astype(str).str.lower()
        == movie_name.lower()
    ].tolist()

    if not movie_indices:
        return []

    movie_index = movie_indices[0]

    # Calculate similarity scores
    distances = list(enumerate(similarity[movie_index]))

    # Sort by similarity score
    distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )

    # Skip the selected movie itself
    distances = distances[1:6]

    recommended_movies = []

    for index, score in distances:

        movie = movies.iloc[index]

        recommended_movies.append({
            "title": movie["title"],
            "movie_id": movie["movie_id"],
            "similarity": float(score),
            "poster_path": movie.get("poster_path", None)
        })

    return recommended_movies


# ==========================================
# MOVIE POSTER FUNCTION
# ==========================================

def get_poster(poster_path):

    if isinstance(poster_path, str) and poster_path.startswith("/"):
        return (
            "https://image.tmdb.org/t/p/w500"
            + poster_path
        )

    return None


# ==========================================
# LOAD APPLICATION DATA
# ==========================================

try:
    movies, similarity = load_data()

except Exception as error:
    st.error(f"Unable to load recommendation model: {error}")
    st.info(
        "Please place movie_list.pkl and similarity.pkl "
        "inside the artifacts folder."
    )
    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎬 CineMatch")

    st.markdown("---")

    st.subheader("About the Project")

    st.write(
        """
        CineMatch is a content-based movie
        recommendation system.

        It recommends movies based on:

        - Movie Genres
        - Keywords
        - Cast Members
        - Directors
        - Movie Overview
        """
    )

    st.markdown("---")

    st.subheader("Technologies")

    st.write("🐍 Python")
    st.write("📊 Pandas & NumPy")
    st.write("🤖 Scikit-learn")
    st.write("🧠 NLP")
    st.write("📐 Cosine Similarity")
    st.write("🎨 Streamlit")

    st.markdown("---")

    st.caption("Built using the TMDb 5000 Movie Dataset")


# ==========================================
# MAIN HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover your next favorite movie with AI-powered recommendations'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# MOVIE SELECTION
# ==========================================

st.markdown("### 🔍 Find Similar Movies")

movie_list = sorted(
    movies["title"].dropna().astype(str).unique().tolist()
)

selected_movie = st.selectbox(
    "Choose a movie you like:",
    movie_list,
    index=None,
    placeholder="Search or select a movie..."
)

recommend_button = st.button(
    "✨ Recommend Movies",
    use_container_width=True
)


# ==========================================
# DISPLAY SELECTED MOVIE
# ==========================================

if recommend_button:

    if selected_movie is None:

        st.warning("Please select a movie first.")

    else:

        recommendations = recommend(
            selected_movie,
            movies,
            similarity
        )

        if recommendations:

            st.markdown(
                '<div class="section-title">🎥 Movies You May Like</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"Because you liked **{selected_movie}**, "
                "here are five similar movies:"
            )

            # Display recommended movies
            columns = st.columns(5)

            for i, movie in enumerate(recommendations):

                with columns[i]:

                    poster_url = get_poster(
                        movie["poster_path"]
                    )

                    if poster_url:

                        st.image(
                            poster_url,
                            use_container_width=True
                        )

                    else:

                        st.markdown(
                            """
                            <div style="
                                height: 250px;
                                background-color: #29293d;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                border-radius: 10px;
                                color: #aaaaaa;
                                font-size: 40px;
                            ">
                                🎬
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f'<div class="movie-title">'
                        f'{movie["title"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.caption(
                        f"Similarity: {movie['similarity'] * 100:.1f}%"
                    )

                    st.caption(
                        f"TMDb ID: {movie['movie_id']}"
                    )

        else:

            st.warning(
                "No recommendations found for this movie."
            )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#999999;">
        🎬 CineMatch | Content-Based Movie Recommendation System
        <br>
        Powered by Python, NLP and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
