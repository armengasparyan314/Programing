import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Էջի լայն կարգավորում
st.set_page_config(page_title="Premier League Clustering Dashboard", layout="wide")

st.title("⚽ Premier League Player Clustering & Analytics (2025-26)")
st.write("This interactive web application groups Premier League players using unsupervised machine learning (K-Means & PCA).")

# 1. Տվյալների բեռնման և մշակման ֆունկցիա (Cache-ավորված)
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('premier_league_complete_stats_until31thGameDayOnSeason2025-26.csv', encoding='latin-1')
    
    drop_cols = ['team_name', 'bigChancesMissed', 'cleanSheet', 'clearances', 'dispossessed', 'dribbledPast', 'errorLeadToGoal', 'errorLeadToShot',
                 'freeKickGoal',  'goalsPrevented', 'headedGoals', 'highClaims', 'hitWoodwork', 'leftFootGoals', 'savesParried',
                 'offsides', 'ownGoals', 'passToAssist', 'penaltiesTaken', 'penaltyConceded', 'penaltyConversion', 'setPieceConversion',
                 'penaltyFaced', 'penaltyGoals', 'penaltySave', 'penaltyWon', 'possessionWonAttThird', 'punches', 'shotFromSetPiece',
                 'redCards', 'rightFootGoals', 'savedShotsFromInsideTheBox', 'savedShotsFromOutsideTheBox', 'saves', 'savesCaught', 
                 'shotsFromOutsideTheBox', 'shotsOffTarget', 'shotsOnTarget']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    features = ['accurateCrosses', 'accurateFinalThirdPasses', 'accurateLongBalls',  'accuratePasses', 'aerialDuelsWon', 'assists', 'ballRecovery',
                'bigChancesCreated', 'expectedAssists', 'expectedGoals', 'goals', 'goalsAssistsSum', 'groundDuelsWon', 'keyPasses', 'successfulDribbles',
                'tacklesWon', 'totalCross', 'totalDuelsWon',  'totalPasses', 'totalShots', 
                'yellowCards', 'touches', 'accurateOwnHalfPasses']
    
    x = df[features].fillna(df[features].median())

    if 'position' in df.columns:
        df['position'] = LabelEncoder().fit_transform(df['position'])

    df['expectedGoals'] = df['expectedGoals'].fillna(df['expectedGoals'].median())
    df['expectedAssists'] = df['expectedAssists'].fillna(df['expectedAssists'].median())

    x_scaled = StandardScaler().fit_transform(x)
    return df, x, x_scaled, features

df, x, x_scaled, features = load_and_prep_data()

# --- ԲԱԺԻՆ 1: EDA (Հարաբերկցության մատրիցաներ) ---
st.header("📊 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

key_columns_attack = ['accurateCrosses', 'accurateFinalThirdPasses', 'assists', 'expectedAssists',
                      'bigChancesCreated', 'goals', 'expectedGoals', 'goalsAssistsSum', 'keyPasses', 'successfulDribbles']

key_columns_def = ['accuratePasses', 'totalPasses', 'aerialDuelsWon', 'ballRecovery', 'groundDuelsWon', 
                   'tacklesWon', 'totalDuelsWon', 'yellowCards', 'accurateOwnHalfPasses', 'touches']

with col1:
    st.subheader("Attacking Skills Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(x[key_columns_attack].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax, annot_kws={"size": 8})
    st.pyplot(fig)

with col2:
    st.subheader("Defending Skills Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(x[key_columns_def].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax, annot_kws={"size": 8})
    st.pyplot(fig)


# --- ԲԱԺԻՆ 2: Machine Learning & K-Means ---
st.header("🤖 Machine Learning & Clustering")

# Կողային պանել K-ի ընտրության համար
k_value = st.sidebar.slider("Select Number of Clusters (K)", min_value=3, max_value=6, value=4)

# KMeans և PCA հաշվարկ
model = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
df['Cluster_Raw'] = model.fit_predict(x_scaled)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
df['PCA1'] = x_pca[:, 0]
df['PCA2'] = x_pca[:, 1]

# Silhouette Score
score = silhouette_score(x_scaled, df['Cluster_Raw'])
st.info(f"**Silhouette Score for K = {k_value}:** `{score:.4f}`")

# Գրաֆիկ՝ հայտնի խաղացողների նշումով (ինչպես քո կոդում էր)
st.subheader(f"Player Clusters Visualization (K = {k_value})")

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster_Raw'], cmap='viridis', alpha=0.7)

# Հայտնի խաղացողների կոորդինատների նշում
players_to_highlight = {
    'Elliot Anderson': 'orange',
    'Igor Thiago': 'black',
    'Erling Haaland': 'blue',
    'Bruno Fernandes': 'red',
    'Hugo Ekitiké': 'purple'
}

for p_name, color in players_to_highlight.items():
    p_data = df[df['player_name'] == p_name]
    if not p_data.empty:
        px, py = p_data['PCA1'].values[0], p_data['PCA2'].values[0]
        ax.scatter(px, py, color=color, s=120, edgecolors='white', zorder=5)
        ax.text(px, py, f"  {p_name}", fontsize=9, fontweight='bold', color='black')

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)


# --- ԲԱԺԻՆ 3: Cluster Summary (Միջին արժեքներ) ---
st.header("📋 Cluster Profiles (Mean Values)")
cluster_summary = df.groupby('Cluster_Raw')[features[:10]].mean()
st.dataframe(cluster_summary)