"""Command-line interface for credit card customer segmentation."""
import click
import pandas as pd
from pathlib import Path
from . import (
    load_customer_data,
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

@click.group()
def cli():
    """Credit Card Customer Segmentation CLI."""
    pass

@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--n-clusters', default=8, help='Number of clusters to create')
@click.option('--output-dir', default='outputs', help='Directory to save outputs')
def analyze(data_path: str, n_clusters: int, output_dir: str):
    """Perform customer segmentation analysis.
    
    Args:
        data_path: Path to the CSV file containing customer data
        n_clusters: Number of clusters to create
        output_dir: Directory to save outputs
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load and prepare data
    click.echo("Loading and preparing data...")
    df = load_customer_data(data_path)
    df_prepared = prepare_features(df)
    
    # Find optimal clusters
    click.echo("Finding optimal number of clusters...")
    inertias = find_optimal_clusters(df_prepared.values, max_clusters=15)
    
    # Set plotting style
    set_plotting_style()
    
    # Plot and save elbow curve
    click.echo("Generating elbow curve...")
    elbow_fig = plot_elbow_curve(inertias)
    elbow_fig.savefig(output_path / 'elbow_curve.png')
    
    # Perform clustering
    click.echo(f"Performing clustering with {n_clusters} clusters...")
    labels, model = perform_clustering(df_prepared.values, n_clusters=n_clusters)
    
    # Add cluster labels to original dataframe
    df['CLUSTER'] = labels + 1
    
    # Generate and save visualizations
    click.echo("Generating visualizations...")
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                   if col not in ['customer_id', 'CLUSTER']]
    
    dist_fig = plot_cluster_distributions(df, numeric_cols)
    dist_fig.savefig(output_path / 'cluster_distributions.png')
    
    rel_fig = plot_cluster_relationships(df)
    rel_fig.savefig(output_path / 'cluster_relationships.png')
    
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_fig = plot_categorical_distributions(df, cat_cols)
    cat_fig.savefig(output_path / 'categorical_distributions.png')
    
    # Generate cluster statistics
    click.echo("Calculating cluster statistics...")
    stats = get_cluster_statistics(df, labels)
    stats.to_csv(output_path / 'cluster_statistics.csv')
    
    # Save clustered data
    df.to_csv(output_path / 'clustered_data.csv', index=False)
    
    click.echo(f"Analysis complete! Results saved to {output_path}")

if __name__ == '__main__':
    cli()