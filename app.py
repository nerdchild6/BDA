import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
from scipy.stats import zscore
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Step 1: Load Dataset
st.title("Project2_Group02")

# We assume the user has uploaded the file to the same directory or it's provided.
csv_path = "marketing_campaign.csv"
if os.path.exists(csv_path):
    # Load the data
    df = pd.read_csv(csv_path, sep='\t')
    st.write("Customer Personality Analysis")
    
    # Show the dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Data Preprocessing
    st.subheader("Data Preprocessing")
    st.write("Cleaning and dropping unnecessary columns...")
    df_cleaned = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'Education', 'Marital_Status'])
    
    # Handling missing values and duplicates
    df_cleaned.drop_duplicates(inplace=True)
    df_cleaned.dropna(inplace=True)
    
    # Handling outliers
    cols = [
        'Recency', 'Income', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
        'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    
    mask = pd.Series(True, index=df_cleaned.index)
    for col in cols:
        z_scores = zscore(df_cleaned[col].dropna())
        z_scores = pd.Series(z_scores, index=df_cleaned[col].dropna().index)
        mask &= z_scores.abs() <= 3
    
    df_cleaned = df_cleaned.loc[mask]
    
    # Show cleaned dataset preview
    st.write("Cleaned Data:")
    st.write(df_cleaned.head())
    
    # Feature scaling
    st.write("Scaling the features using MinMaxScaler...")
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_cleaned[numeric_cols])
    df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols, index=df_cleaned.index)

    # --- START OF MODIFICATION: Cluster the full dataset first ---
    st.subheader("Clustering the Full Dataset")
    st.write("Clustering on all scaled numeric features to get consistent segments.")
    
    # Perform K-means on all scaled numeric features
    n_clusters = 4
    kmeans_full = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    df_cleaned['Cluster'] = kmeans_full.fit_predict(df_scaled)
    # --- END OF MODIFICATION ---

    # PCA Visualization
    st.subheader("PCA Analysis")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_cleaned_pca = df_cleaned.copy()
    df_cleaned_pca['PC1'] = df_pca['PC1']
    df_cleaned_pca['PC2'] = df_pca['PC2']
    
    # Define a fixed color map for each cluster label, used for all visualizations
    color_map = {
        0: 'green',
        1: 'purple',
        2: 'red',
        3: 'blue'
    }
    colors_pca = df_cleaned_pca['Cluster'].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df_cleaned_pca['PC1'], df_cleaned_pca['PC2'], c=colors_pca, alpha=0.5)
    ax.set_title("PCA - 2D Projection with Fixed Cluster Colors")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)
    
    # Clustering with KMeans Visualization
    st.subheader("KMeans Clustering Visualization")
    
    # Let user select features for x and y axes
    feature_options = list(df_cleaned.columns)
    # Filter out 'Cluster' from feature options
    if 'Cluster' in feature_options:
        feature_options.remove('Cluster')
    
    x_feature = st.selectbox("Select feature for X-axis", feature_options, index=feature_options.index('Income') if 'Income' in feature_options else 0)
    y_feature = st.selectbox("Select feature for Y-axis", feature_options, index=feature_options.index('MntWines') if 'MntWines' in feature_options else 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_scatter = df_cleaned['Cluster'].map(color_map)
    
    scatter = ax.scatter(df_cleaned[x_feature], df_cleaned[y_feature], c=colors_scatter, alpha=0.6)
    ax.set_title(f"Customer Segmentation based on {x_feature} and {y_feature}")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    
    # Add centroids for the current selected features
    centroids_df = pd.DataFrame(kmeans_full.cluster_centers_, columns=numeric_cols)
    ax.scatter(centroids_df[x_feature], centroids_df[y_feature], s=500, c='black', marker='X', label='Centroids')
    ax.legend()
    
    st.pyplot(fig)

    # Evaluation metrics
    silhouette = silhouette_score(df_scaled, df_cleaned['Cluster'])
    calinski_harabasz = calinski_harabasz_score(df_scaled, df_cleaned['Cluster'])
    davies_bouldin = davies_bouldin_score(df_scaled, df_cleaned['Cluster'])

    st.write(f"Silhouette Score (on full dataset): {silhouette:.3f}")
    st.write(f"Calinski-Harabasz Index (on full dataset): {calinski_harabasz:.3f}")
    st.write(f"Davies-Bouldin Index (on full dataset): {davies_bouldin:.3f}")
