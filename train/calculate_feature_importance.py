import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRanker, Pool
from MSA_train import MSAConfig, MSAPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_best_model(models_dir):
    """Load the best trained model from disk."""
    candidate_files = [f for f in os.listdir(models_dir) if f.endswith('_best_ranker.cbm')]
    if not candidate_files:
        raise FileNotFoundError(
            f"No *_best_ranker.cbm model file found in {models_dir}. "
            "Please run training first."
        )
    candidate_paths = [os.path.join(models_dir, f) for f in candidate_files]
    best_path = max(candidate_paths, key=os.path.getmtime)
    logger.info(f"Loading best model from: {best_path}")
    model = CatBoostRanker()
    model.load_model(best_path)
    return model, best_path

def calculate_feature_importance(model, feature_names, train_data, output_dir):
    """Calculate and save feature importance."""
    X = train_data[feature_names].values
    y = train_data['hit'].values
    group_id = train_data['Enzyme1'].astype(int).values
    
    train_pool = Pool(data=X, label=y, group_id=group_id)
    feature_importance = model.get_feature_importance(train_pool)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved: {importance_path}")
    
    logger.info("Feature importance values:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return importance_df

def visualize_feature_importance(importance_df, output_dir):
    """Create visualizations for feature importance."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    features = importance_df['Feature'].values
    importance_values = importance_df['Importance'].values
    
    # Bar plot
    ax1 = axes[0]
    bars = ax1.barh(features, importance_values, color=['#2E86AB', '#A23B72'])
    ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, importance_values)):
        ax1.text(val + max(importance_values) * 0.01, i, f'{val:.2f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Pie chart
    ax2 = axes[1]
    colors = ['#2E86AB', '#A23B72']
    wedges, texts, autotexts = ax2.pie(importance_values, labels=features, 
                                       autopct='%1.1f%%', colors=colors,
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Feature Importance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature importance plot saved: {plot_path}")
    
    plt.close()

def main():
    """Main function to calculate and visualize feature importance."""
    logger.info("="*60)
    logger.info("Feature Importance Calculation and Visualization")
    logger.info("="*60)
    
    config = MSAConfig()
    feature_names = MSAPipeline.RANKER_FEATURES
    
    try:
        model, model_path = load_best_model(config.models_dir)
        logger.info(f"Model loaded successfully from: {os.path.basename(model_path)}")
        
        logger.info("Loading training data...")
        train_data_path = os.path.join(config.user_dir, 'train_final_data.csv')
        if not os.path.exists(train_data_path):
            logger.info("Training data not found. Preparing data...")
            pipeline = MSAPipeline(config)
            pipeline.prepare_data()
            train_data = pipeline.train_data
        else:
            train_data = pd.read_csv(train_data_path)
            logger.info(f"Training data loaded: {len(train_data)} samples")
        
        if train_data is None or len(train_data) == 0:
            raise ValueError("Training data is empty. Please run data preparation first.")
        
        importance_df = calculate_feature_importance(model, feature_names, train_data, config.models_dir)
        
        visualize_feature_importance(importance_df, config.models_dir)
        
        logger.info("="*60)
        logger.info("Feature importance calculation completed successfully!")
        logger.info("="*60)
        
        return importance_df
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please run training first using: python MSA_train.py or python run_grid_search.py")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

