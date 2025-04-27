"""Credit Card Customer Segmentation package."""
from credit_card_segmentation.utils.data_loader import (
    load_customer_data,
    get_numeric_features,
    get_categorical_features
)
from credit_card_segmentation.src.feature_engineering import prepare_features
from credit_card_segmentation.src.clustering import (
    find_optimal_clusters,
    perform_clustering,
    get_cluster_statistics
)
from credit_card_segmentation.utils.plotting import (
    set_plotting_style,
    plot_cluster_distributions,
    plot_cluster_relationships,
    plot_categorical_distributions,
    plot_elbow_curve
)

__version__ = '0.1.0'
__all__ = [
    'load_customer_data',
    'get_numeric_features',
    'get_categorical_features',
    'prepare_features',
    'find_optimal_clusters',
    'perform_clustering',
    'get_cluster_statistics',
    'set_plotting_style',
    'plot_cluster_distributions',
    'plot_cluster_relationships',
    'plot_categorical_distributions',
    'plot_elbow_curve'
]