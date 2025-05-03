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
st.title("Customer Personality Analysis")

csv_path = "marketing_campaign.csv"
if os.path.exists(csv_path):
    # Load the data
    df = pd.read_csv(csv_path, sep='\t')
    st.write("Dataset Loaded")
    
    # Show the dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Data Preprocessing
    st.subheader("Data Preprocessing")
    st.write("Cleaning and dropping unnecessary columns...")
    df_cleaned = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'])
    
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
    
    df_cleaned = df_cleaned.loc[mask]  # Apply mask to df_cleaned instead of df
    
    # Show cleaned dataset preview
    st.write("Cleaned Data:")
    st.write(df_cleaned.head())
    
    # Feature scaling
    st.write("Scaling the features using MinMaxScaler...")
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
    
    # PCA Visualization
    st.subheader("PCA Analysis")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_cleaned[numeric_cols])
    
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    df_cleaned_pca = df_cleaned.copy()
    df_cleaned_pca['PC1'] = df_pca['PC1']
    df_cleaned_pca['PC2'] = df_pca['PC2']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df_cleaned_pca['PC1'], df_cleaned_pca['PC2'], c=df_cleaned_pca['PC1'], cmap='viridis', alpha=0.5)
    plt.colorbar(sc, label='PC1')
    ax.set_title("PCA - 2D Projection with Color-coded PC1")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)
    
    # Clustering with KMeans
    st.subheader("KMeans Clustering")
    X = df_cleaned[['Income', 'MntWines']].dropna()
    kmeans = KMeans(n_clusters=4, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    
    df_cleaned['Cluster'] = y_kmeans
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_cleaned['Income'], df_cleaned['MntWines'], c=df_cleaned['Cluster'], cmap='viridis', alpha=0.6)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=500, c='red', marker='X', label='Centroids')
    ax.set_title("Customer Segmentation based on Income and Spending on MntWines")
    ax.set_xlabel("Income")
    ax.set_ylabel("MntWines")
    ax.legend()
    st.pyplot(fig)
    
    # Evaluation metrics
    silhouette = silhouette_score(X, y_kmeans)
    calinski_harabasz = calinski_harabasz_score(X, y_kmeans)
    davies_bouldin = davies_bouldin_score(X, y_kmeans)
    
    st.write(f"Silhouette Score: {silhouette:.3f}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin:.3f}")



