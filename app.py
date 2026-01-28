import streamlit as st
import numpy as np
import pandas as pd
import pickle
import h5py

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Anime Recommendation System",
    layout="wide"
)

st.title("🎌 Anime Recommendation System")

# -------------------------
# LOAD MODEL WEIGHTS
# -------------------------
@st.cache_resource
def extract_weights(file_path, layer_name):
    with h5py.File(file_path, 'r') as h5_file:
        if layer_name in h5_file:
            weight_layer = h5_file[layer_name]
            weights = weight_layer[()]
            # Normalize each row
            weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
            return weights
    raise KeyError(f"Layer {layer_name} not found in {file_path}")

anime_weights = extract_weights(
    "model/myanimeweights.h5",
    "anime_embedding/anime_embedding/embeddings:0"
)

user_weights = extract_weights(
    "model/myanimeweights.h5",
    "user_embedding/user_embedding/embeddings:0"
)

# -------------------------
# LOAD ENCODERS & DATA
# -------------------------
@st.cache_resource
def load_data():
    with open("model/anime_encoder.pkl", "rb") as f:
        anime_encoder = pickle.load(f)

    with open("model/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)

    with open("model/anime-dataset-2023.pkl", "rb") as f:
        df_anime = pickle.load(f)

    df_users = pd.read_csv("model/users-score-2023.csv", low_memory=True)
    df_anime = df_anime.replace("UNKNOWN", "")

    return anime_encoder, user_encoder, df_anime, df_users

anime_encoder, user_encoder, df_anime, df_users = load_data()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def display_anime(row):
    """Display an anime recommendation nicely in Streamlit."""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(row.get("Image URL", ""), width=120)
    with col2:
        st.markdown(f"### {row.get('Name', '')}")
        if "Similarity" in row:
            st.write(f"**Similarity:** {row['Similarity']}")
        st.write(f"**Genres:** {row.get('Genres', '')}")
        st.write(f"**Score:** {row.get('Score', '')}")
        if "Synopsis" in row:
            st.write(row.get("Synopsis", ""))

# -------------------------
# USER-BASED FUNCTIONS
# -------------------------
def find_similar_users(user_id, n=15):
    """Find top n similar users based on embedding similarity."""
    try:
        encoded_index = user_encoder.transform([user_id])[0]
    except ValueError:
        return pd.DataFrame()
    
    dists = np.dot(user_weights, user_weights[encoded_index])
    closest = np.argsort(dists)[-n:][::-1]  # top n
    data = []
    for idx in closest:
        decoded = user_encoder.inverse_transform([idx])[0]
        data.append({
            "similar_users": decoded,
            "similarity": dists[idx]
        })
    return pd.DataFrame(data).sort_values(by="similarity", ascending=False)

def get_user_preferences(user_id):
    """Get top-rated animes for a specific user."""
    watched = df_users[df_users["user_id"] == user_id]
    if watched.empty:
        return pd.DataFrame()
    
    if len(watched) < 3:  # fallback for few ratings
        top_ids = watched.anime_id.values
    else:
        threshold = np.percentile(watched.rating, 75)
        top_ids = watched[watched.rating >= threshold].anime_id.values

    return df_anime[df_anime.anime_id.isin(top_ids)][["Name", "Genres"]]

def get_recommended_animes(similar_users, user_pref, n=10, genres=[]):
    """Generate recommendations from similar users excluding already watched animes."""
    anime_pool = []
    for uid in similar_users.similar_users:
        pref = get_user_preferences(int(uid))
        if not pref.empty:
            pref = pref[~pref.Name.isin(user_pref.Name)]
            if genres:
                pref = pref[pref.Genres.apply(lambda x: any(g in x for g in genres))]
            anime_pool.extend(pref.Name.values)

    if not anime_pool:
        return pd.DataFrame()

    top = pd.Series(anime_pool).value_counts().head(n).index
    df_top = df_anime[df_anime.Name.isin(top)][[
        "Image URL", "Name", "Genres", "Score", "Synopsis"
    ]]
    if genres:
        df_top = df_top[df_top.Genres.apply(lambda x: any(g in x for g in genres))]
    return df_top

# -------------------------
# ITEM-BASED FUNCTIONS
# -------------------------
def find_similar_animes(name, n=10, genres=[]):
    """Find top n similar animes based on embedding similarity and optional genres."""
    row = df_anime[df_anime.Name == name]
    if row.empty:
        return pd.DataFrame()

    anime_id = row.anime_id.values[0]
    encoded = anime_encoder.transform([anime_id])[0]
    dists = np.dot(anime_weights, anime_weights[encoded])
    closest = np.argsort(dists)[-n-1:][::-1]

    results = []
    for idx in closest:
        decoded = anime_encoder.inverse_transform([idx])[0]
        anime = df_anime[df_anime.anime_id == decoded]
        if not anime.empty:
            # Genre filtering
            if genres and not any(g in anime.Genres.values[0] for g in genres):
                continue
            results.append({
                "Name": anime.Name.values[0],
                "Similarity": f"{dists[idx]*100:.2f}%",
                "Genres": anime.Genres.values[0],
                "Score": anime.Score.values[0],
                "Image URL": anime["Image URL"].values[0]
            })

    df_res = pd.DataFrame(results)
    return df_res[df_res.Name != name]

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("⚙️ Settings")

rec_type = st.sidebar.selectbox(
    "Recommendation Type",
    ["User-Based", "Item-Based"]
)

num_recs = st.sidebar.slider(
    "Number of Recommendations",
    5, 20, 10
)

# Genre filter
all_genres = sorted({g.strip() for gs in df_anime.Genres.dropna() for g in gs.split(",")})

# Dynamic genre filter based on recommendation type
st.sidebar.subheader("Filter by Genre (optional)")
use_all_genres = st.sidebar.checkbox("Use all genres", value=True)
if use_all_genres:
    selected_genres = []
else:
    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        all_genres,
        default=[]
    )

# -------------------------
# USER-BASED UI
# -------------------------
if rec_type == "User-Based":
    st.subheader("👤 User-Based Recommendation")
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend"):
        similar_users = find_similar_users(user_id)
        if similar_users.empty:
            st.warning("User ID not found or no similar users available.")
        else:
            similar_users = similar_users[similar_users.similarity > 0.4]
            user_pref = get_user_preferences(user_id)
            recs = get_recommended_animes(similar_users, user_pref, num_recs, genres=selected_genres)

            if recs.empty:
                st.warning("No recommendations found for this user with selected genre(s).")
            else:
                for _, row in recs.iterrows():
                    display_anime(row)

# -------------------------
# ITEM-BASED UI
# -------------------------
else:
    st.subheader("🎬 Item-Based Recommendation")
    anime_name = st.text_input(
        "Enter Anime Name",
        placeholder="Type anime name..."
    )

    if st.button("Recommend"):
        recs = find_similar_animes(anime_name, num_recs, genres=selected_genres)
        if recs.empty:
            st.warning("Anime not found or no similar animes available with selected genre(s).")
        else:
            for _, row in recs.iterrows():
                display_anime(row)
