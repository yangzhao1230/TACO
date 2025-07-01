from dataclasses import dataclass
import scanpy as sc
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

@dataclass
class Arguments:
    cell: str = "complex"
    train_rounds: int = 100000
    early_stopping_rounds: int = 500  # Early stopping after 100 rounds with no improvement
    random_state: int = 42  # Random seed for reproducibility

def save_checkpoint(model, filename):
    """Save model checkpoint."""
    model.save_model(filename)

def main():
    args = Arguments()
    print(f"Training model for {args.cell}...")
    # Load the pre-split datasets
    train_data_path = f"/home/v-zhaoyang2/project/DNADesign/regLM/yeast_promoters/01_data_processing/motif_data/surrogate_{args.cell}_train_500000_sites.h5ad"
    val_data_path = f"/home/v-zhaoyang2/project/DNADesign/regLM/yeast_promoters/01_data_processing/motif_data/surrogate_{args.cell}_val_500000_sites.h5ad"
    test_data_path = f"/home/v-zhaoyang2/project/DNADesign/regLM/yeast_promoters/01_data_processing/motif_data/surrogate_{args.cell}_test_500000_sites.h5ad"

    train_adata = sc.read_h5ad(train_data_path)
    val_adata = sc.read_h5ad(val_data_path)
    test_adata = sc.read_h5ad(test_data_path)

    # Extract features (X) and labels (y) for training, validation, and testing
    X_train, y_train = train_adata.X, train_adata.obs[f'exp_{args.cell}'].values
    X_valid, y_valid = val_adata.X, val_adata.obs[f'exp_{args.cell}'].values
    X_test, y_test = test_adata.X, test_adata.obs[f'exp_{args.cell}'].values

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"y_valid shape: {y_valid.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'seed': args.random_state
    }

    # Training with early stopping and logging
    print(f"Training model for {args.cell}...")
    early_stopping_callback = lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True)
    log_evaluation_callback = lgb.log_evaluation(period=50)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.train_rounds,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[early_stopping_callback, log_evaluation_callback]
    )

    # Save the best model
    save_checkpoint(model, f'saved/model_{args.cell}_mbo_best.txt')
    print(f"Model saved to 'saved/model_{args.cell}_mbo_best.txt'")

    # Evaluate on the validation set
    y_valid_pred = model.predict(X_valid)
    rmse_valid = mean_squared_error(y_valid, y_valid_pred, squared=False)
    pearson_valid = pearsonr(y_valid, y_valid_pred)[0]
    spearman_valid = spearmanr(y_valid, y_valid_pred)[0]

    print(f"Validation RMSE for {args.cell}: {rmse_valid}")
    print(f"Validation Pearson correlation for {args.cell}: {pearson_valid}")
    print(f"Validation Spearman correlation for {args.cell}: {spearman_valid}")

    # Evaluate on the test set
    y_test_pred = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    pearson_test = pearsonr(y_test, y_test_pred)[0]
    spearman_test = spearmanr(y_test, y_test_pred)[0]

    print(f"Test RMSE for {args.cell}: {rmse_test}")
    print(f"Test Pearson correlation for {args.cell}: {pearson_test}")
    print(f"Test Spearman correlation for {args.cell}: {spearman_test}")

if __name__ == "__main__":
    main()
