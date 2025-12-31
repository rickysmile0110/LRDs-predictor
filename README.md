# CYP-predictor

A machine learning-based predictor for cytochrome P450 (CYP) enzyme-substrate interactions using Multiple Sequence Alignment (MSA) and CatBoost Ranker.

## Overview

This tool predicts potential substrate-enzyme interactions for cytochrome P450 enzymes by combining:
- Multiple Sequence Alignment (MSA) for enzyme similarity analysis
- Substrate similarity using PCA-based features
- CatBoost Ranker for learning-to-rank predictions

## Features

- Automated MSA-based enzyme similarity calculation
- Substrate neighbor identification using k-nearest neighbors
- Hyperparameter grid search for optimal model performance
- Comprehensive evaluation metrics (NDCG, Precision, Recall, Enrichment)
- Feature importance analysis

## Installation

### Prerequisites

- Python 3.7+
- Clustal Omega (for MSA alignment)

### Install Clustal Omega

Clustal Omega is required for Multiple Sequence Alignment (MSA). Please refer to the [official Clustal Omega website](https://www.clustal.org/omega/) for detailed installation instructions.

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
CYPs-predictor/
├── train/                          # Training pipeline
│   ├── MSA_train.py                # Main training script
│   ├── run_grid_search.py          # Run hyperparameter grid search
│   ├── run_evaluate.py             # Run evaluation only
│   ├── calculate_feature_importance.py  # Feature importance analysis
│   ├── DataPrep/                   # Training data files
│   │   ├── Reported Sequences.fasta
│   │   ├── Reaction Table.csv
│   │   ├── Substrates_PCs.csv
│   │   └── Reported ID.csv
│   ├── UserData/                   # Generated intermediate data
│   └── MSA_Models/                 # Trained models and results
├── prediction/                     # Prediction pipeline
│   ├── MSA_prediction.py           # Main prediction script
│   ├── run_grid_search.py          # Run grid search for prediction
│   ├── run_evaluate.py             # Run prediction only
│   ├── calculate_feature_importance.py  # Feature importance analysis
│   ├── DataPrep/                   # Input data files
│   │   ├── Reported Sequences.fasta
│   │   ├── Ef Sequences.fasta     # Test sequences
│   │   ├── Reaction Table.csv
│   │   ├── Substrates_PCs.csv
│   │   └── Reported ID.csv
│   ├── UserData/                   # Generated intermediate data
│   └── MSA_Models/                 # Prediction results
└── README.md
```

## Usage

### Training

1. Prepare your data files in the `train/DataPrep/` directory:
   - `Reported Sequences.fasta` - Training enzyme sequences in FASTA format
   - `Reaction Table.csv` - Known enzyme-substrate interactions
   - `Substrates_PCs.csv` - Substrate PCA features
   - `Reported ID.csv` - Enzyme IDs

2. Run complete training pipeline:
```bash
cd train
python MSA_train.py
```

Or run grid search only:
```bash
python run_grid_search.py
```

Or run evaluation only (requires pre-trained models):
```bash
python run_evaluate.py
```

### Prediction

1. Prepare test data in `prediction/DataPrep/` directory:
   - `Ef Sequences.fasta` - Test enzyme sequences
   - Same data files as training

2. Run complete prediction pipeline:
```bash
cd prediction
python MSA_prediction.py
```

Or run prediction only (requires pre-trained models):
```bash
python run_evaluate.py
```

### Feature Importance Analysis

After training or prediction, analyze feature importance:

**For training models:**
```bash
cd train
python calculate_feature_importance.py
```

**For prediction models:**
```bash
cd prediction
python calculate_feature_importance.py
```

## Configuration

Key parameters can be adjusted in `MSAConfig` class:

- `top_k`: Number of top similar enzymes to consider (default: 10)
- `pca_components`: Number of PCA components for substrates (default: 5)
- `num_neighbors`: Number of substrate neighbors (default: 10)
- `target_neighbors`: Number of neighbors to use (default: 9)
- `random_state`: Random seed for reproducibility (default: 42)
- `clustal_exe`: Path to Clustal Omega executable (default: 'clustalo', can be set via `CLUSTALO_PATH` environment variable)

## Output Files

### Training Outputs (`train/MSA_Models/`)

- `iters{iterations}_depth{depth}.cbm` - Individual trained models
- `iters{iterations}_depth{depth}_best_ranker.cbm` - Best model from grid search
- `msa_grid_search_results.csv` - Grid search results with NDCG@20 scores
- `train_predictions.csv` - Training set predictions
- `test_predictions.csv` - Test set predictions
- `metrics_comparison.csv` - Comprehensive evaluation metrics
- `feature_importance.csv` - Feature importance values
- `feature_importance_plot.png` - Feature importance visualization

### Prediction Outputs (`prediction/MSA_Models/`)

- `predictions.csv` - Final predictions for test enzymes
- `feature_importance.csv` - Feature importance values (if feature importance analysis is run)
- `feature_importance_plot.png` - Feature importance visualization (if feature importance analysis is run)

## Evaluation Metrics

The model is evaluated using:
- **NDCG@k**: Normalized Discounted Cumulative Gain at top k
- **Precision@k**: Precision at top k predictions
- **Recall@k**: Recall at top k predictions
- **Enrichment@k**: Enrichment factor at top k
- **Rank of first hit**: Position of first true positive

## Citation

This work is currently under review. Citation information will be updated upon publication.

## Related Work

The overall architecture and methodology of this tool is inspired by:

Paton, A.E., Boiko, D.A., Perkins, J.C. et al. Connecting chemical and protein sequence space to predict biocatalytic reactions. *Nature* *646*, 108–116 (2025). https://doi.org/10.1038/s41586-025-09519-5

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub.

