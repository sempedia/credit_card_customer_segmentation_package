# Credit Card Customer Segmentation

A Python package for customer segmentation analysis using K-means clustering. This package helps analyze credit card customer behavior and segment customers into meaningful groups based on their characteristics and usage patterns.

## Features

- Data preprocessing and feature engineering
- Automatic feature scaling and encoding
- K-means clustering with optimal cluster selection
- Rich visualizations of cluster characteristics
- Detailed cluster statistics and analysis
- Command-line interface for easy use
- Comprehensive test suite

## Installation

### Using Poetry (Recommended)

1. Install Poetry if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and install:

```bash
git clone https://github.com/sempedia/credit_card_customer_segmentation.git
cd credit_card_customer_segmentation
poetry install
```

### Using Pip

```bash
pip install git+https://github.com/sempedia/credit_card_customer_segmentation.git
```

## Usage

### Command Line Interface

The package provides a convenient CLI for running the analysis:

```bash
# Basic usage with default parameters
credit-card-segmentation analyze customer_data.csv

# Specify number of clusters and output directory
credit-card-segmentation analyze customer_data.csv --n-clusters 6 --output-dir results
```

### Python API

```python
from credit_card_segmentation import (
    load_customer_data,
    prepare_features,
    perform_clustering,
    get_cluster_statistics
)

# Load and prepare data
df = load_customer_data("customer_data.csv")
df_prepared = prepare_features(df)

# Perform clustering
labels, model = perform_clustering(df_prepared, n_clusters=8)

# Get cluster statistics
stats = get_cluster_statistics(df, labels)
print(stats)
```

## Required Data Format

Your input CSV file should contain the following columns:

- customer_id (numeric): Unique identifier for each customer
- gender (categorical): Customer gender (M/F)
- education_level (categorical): Education level
- marital_status (categorical): Marital status
- age (numeric): Customer age
- months_on_book (numeric): Length of relationship with bank
- credit_limit (numeric): Credit card limit
- total_trans_amount (numeric): Total transaction amount
- avg_utilization_ratio (numeric): Average card utilization ratio

## Output

The analysis generates several files in the specified output directory:

- `elbow_curve.png`: Plot showing optimal number of clusters
- `cluster_distributions.png`: Distribution of numeric features across clusters
- `cluster_relationships.png`: Key feature relationships by cluster
- `categorical_distributions.png`: Distribution of categorical variables in clusters
- `cluster_statistics.csv`: Detailed statistics for each cluster
- `clustered_data.csv`: Original data with cluster assignments

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Style

The project uses:

- Black for code formatting
- isort for import sorting
- flake8 for linting

To format code:

```bash
poetry run black credit_card_segmentation tests
poetry run isort credit_card_segmentation tests
```

## License

MIT License - see LICENSE file for details
