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

    # Let user select features for x and y axes
    feature_options = list(df_cleaned.columns)
    x_feature = st.selectbox("Select feature for X-axis", feature_options, index=feature_options.index('Income') if 'Income' in feature_options else 0)
    y_feature = st.selectbox("Select feature for Y-axis", feature_options, index=feature_options.index('MntWines') if 'MntWines' in feature_options else 1)

    X = df_cleaned[[x_feature, y_feature]].dropna()
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    df_cleaned['Cluster'] = y_kmeans

    # --- START OF MODIFICATION ---
    # Define a fixed color map for each cluster label.
    # This ensures consistency and prevents misleading interpretations as features change.
    color_map = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple'
    }

    # Map the cluster labels in the DataFrame to the fixed colors
    colors = df_cleaned['Cluster'].map(color_map)
    # --- END OF MODIFICATION ---

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_cleaned[x_feature], df_cleaned[y_feature], c=colors, alpha=0.6)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=500, c='black', marker='X', label='Centroids')
    ax.set_title(f"Customer Segmentation based on {x_feature} and {y_feature}")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend()
    st.pyplot(fig)

    # Evaluation metrics
    silhouette = silhouette_score(X, y_kmeans)
    calinski_harabasz = calinski_harabasz_score(X, y_kmeans)
    davies_bouldin = davies_bouldin_score(X, y_kmeans)

    st.write(f"Silhouette Score: {silhouette:.3f}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
