import streamlit as st
import numpy as np
import pandas as pd
import pickle
import h5py

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Anime Recommendation System",
    layout="wide"
)

st.title("🎌 Anime Recommendation System")

# ===============================
# LOAD MODEL WEIGHTS
# ===============================
@st.cache_resource
def extract_weights(file_path, layer_name):
    with h5py.File(file_path, 'r') as h5_file:
        weights = h5_file[layer_name][()]
        weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
        return weights


anime_weights = extract_weights(
    "model/myanimeweights.h5",
    "anime_embedding/anime_embedding/embeddings:0"
)

user_weights = extract_weights(
    "model/myanimeweights.h5",
    "user_embedding/user_embedding/embeddings:0"
)

# ===============================
# LOAD ENCODERS + DATASETS (ZIP)
# ===============================
@st.cache_resource
def load_models_and_data():
    with open("model/anime_encoder.pkl", "rb") as f:
        anime_encoder = pickle.load(f)

    with open("model/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)

    # 🔥 LOAD FROM ZIP FILES YOU SENT
    df_anime = pd.read_csv("/mnt/data/anime-dataset-2023.csv.zip")
    df_users = pd.read_csv("/mnt/data/user-filtered.csv.zip", low_memory=True)

    df_anime = df_anime.replace("UNKNOWN", "")

    return anime_encoder, user_encoder, df_anime, df_users


anime_encoder, user_encoder, df_anime, df = load_models_and_data()

# ===============================
# FIND SIMILAR USERS
# ===============================
def find_similar_users(user_id, n=15):
    encoded = user_encoder.transform([user_id])[0]
    similarities = np.dot(user_weights, user_weights[encoded])
    closest = np.argsort(similarities)[-n:]

    data = []
    for idx in closest:
        decoded = user_encoder.inverse_transform([idx])[0]
        data.append({
            "similar_users": decoded,
            "similarity": similarities[idx]
        })

    return pd.DataFrame(data).sort_values(by="similarity", ascending=False)

# ===============================
# USER PREFERENCES
# ===============================
def get_user_preferences(user_id):
    watched = df[df["user_id"] == user_id]
    if watched.empty:
        return pd.DataFrame()

    threshold = np.percentile(watched.rating, 75)
    watched = watched[watched.rating >= threshold]

    top_ids = watched.anime_id.values
    return df_anime[df_anime.anime_id.isin(top_ids)][["Name", "Genres"]]

# ===============================
# USER-BASED RECOMMENDATION
# ===============================
def get_recommended_animes(similar_users, user_pref, n=10):
    anime_pool = []

    for uid in similar_users.similar_users:
        pref = get_user_preferences(int(uid))
        if not pref.empty:
            pref = pref[~pref.Name.isin(user_pref.Name)]
            anime_pool.extend(pref.Name.values)

    if not anime_pool:
        return pd.DataFrame()

    top_names = pd.Series(anime_pool).value_counts().head(n).index
    return df_anime[df_anime.Name.isin(top_names)][
        ["Image URL", "Name", "Genres", "Score", "Synopsis"]
    ]

# ===============================
# ITEM-BASED RECOMMENDATION
# ===============================
def find_similar_animes(anime_name, n=10):
    row = df_anime[df_anime.Name == anime_name]
    if row.empty:
        return pd.DataFrame()

    anime_id = row.anime_id.values[0]
    encoded = anime_encoder.transform([anime_id])[0]
    similarities = np.dot(anime_weights, anime_weights[encoded])

    closest = np.argsort(similarities)[-n-1:]
    results = []

    for idx in closest:
        decoded = anime_encoder.inverse_transform([idx])[0]
        anime = df_anime[df_anime.anime_id == decoded]

        if not anime.empty:
            results.append({
                "Name": anime.Name.values[0],
                "Similarity": f"{similarities[idx]*100:.2f}%",
                "Genres": anime.Genres.values[0],
                "Score": anime.Score.values[0],
                "Image URL": anime["Image URL"].values[0]
            })

    df_res = pd.DataFrame(results)
    return df_res[df_res.Name != anime_name]

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("⚙️ Recommendation Settings")

rec_type = st.sidebar.selectbox(
    "Recommendation Type",
    ["User-Based", "Item-Based"]
)

num_recs = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

# ===============================
# USER-BASED UI
# ===============================
if rec_type == "User-Based":
    st.subheader("👤 User-Based Recommendation")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        step=1
    )

    if st.button("Recommend"):
        similar_users = find_similar_users(user_id)
        similar_users = similar_users[similar_users.similarity > 0.4]

        user_pref = get_user_preferences(user_id)
        recs = get_recommended_animes(similar_users, user_pref, num_recs)

        if recs.empty:
            st.warning("No recommendations found for this user.")
        else:
            for _, row in recs.iterrows():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(row["Image URL"], width=120)
                with col2:
                    st.markdown(f"### {row['Name']}")
                    st.write(f"**Genres:** {row['Genres']}")
                    st.write(f"**Score:** {row['Score']}")
                    st.write(row["Synopsis"])

# ===============================
# ITEM-BASED UI
# ===============================
else:
    st.subheader("🎬 Item-Based Recommendation")

    anime_name = st.selectbox(
        "Select Anime",
        sorted(df_anime.Name.dropna().unique())
    )

    if st.button("Recommend"):
        recs = find_similar_animes(anime_name, num_recs)

        if recs.empty:
            st.warning("Anime not found.")
        else:
            for _, row in recs.iterrows():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(row["Image URL"], width=120)
                with col2:
                    st.markdown(f"### {row['Name']}")
                    st.write(f"**Similarity:** {row['Similarity']}")
                    st.write(f"**Genres:** {row['Genres']}")
                    st.write(f"**Score:** {row['Score']}")
