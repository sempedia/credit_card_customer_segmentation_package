"""Example script demonstrating credit card customer segmentation."""
import pandas as pd
from credit_card_segmentation import (
    load_customer_data,
    get_numeric_features,
    get_categorical_features,
    prepare_features,
    find_optimal_clusters,
    perform_clustering,
    get_cluster_statistics,
    set_plotting_style,
    plot_cluster_distributions,
    plot_cluster_relationships,
    plot_categorical_distributions,
    plot_elbow_curve
)

def main():
    # Load and prepare data
    df = load_customer_data('../customer_segmentation.csv')
    
    # Get feature lists
    numeric_cols = get_numeric_features(df)
    cat_cols = get_categorical_features(df)
    
    # Prepare features for clustering
    df_prepared = prepare_features(df)
    
    # Find optimal number of clusters
    inertias = find_optimal_clusters(df_prepared.values, max_clusters=10)
    
    # Plot elbow curve
    set_plotting_style()
    elbow_fig = plot_elbow_curve(inertias)
    elbow_fig.savefig('elbow_curve.png')
    
    # Perform clustering with optimal number of clusters (8 based on analysis)
    labels, model = perform_clustering(df_prepared.values, n_clusters=8)
    
    # Add cluster labels to original dataframe
    df['CLUSTER'] = labels + 1
    
    # Generate and save visualizations
    numeric_cols = [col for col in numeric_cols if col != 'customer_id']
    dist_fig = plot_cluster_distributions(df, numeric_cols)
    dist_fig.savefig('cluster_distributions.png')
    
    rel_fig = plot_cluster_relationships(df)
    rel_fig.savefig('cluster_relationships.png')
    
    cat_fig = plot_categorical_distributions(df, cat_cols)
    cat_fig.savefig('categorical_distributions.png')
    
    # Get cluster statistics
    cluster_stats = get_cluster_statistics(df, labels)
    print("\nCluster Statistics:")
    print(cluster_stats)

if __name__ == '__main__':
    main()