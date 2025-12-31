import os
import logging
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import ndcg_score
from catboost import CatBoostRanker, Pool
from scipy.spatial import cKDTree
import subprocess
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MSAConfig:

    def __init__(self):
        self.user_dir = 'UserData'
        self.prep_dir = 'DataPrep'
        self.models_dir = 'MSA_Models'
        self.reported_sequences = f'{self.prep_dir}/Reported Sequences.fasta'
        self.ef_sequences = f'{self.prep_dir}/Ef Sequences.fasta'
        self.reaction_table = f'{self.prep_dir}/Reaction Table.csv'
        self.substrates_pcs = f'{self.prep_dir}/Substrates_PCs.csv'
        self.reported_id = f'{self.prep_dir}/Reported ID.csv'
        self.clustal_exe = os.getenv('CLUSTALO_PATH', 'clustalo')
        self.top_k = 10
        self.random_state = 42
        self.pca_components = 5
        self.num_neighbors = 10
        self.target_neighbors = 9
        self.total_substrates = 17
        for dir_path in [self.user_dir, self.prep_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)

class MSAPipeline:
    ESSENTIAL_COLUMNS = ['Enzyme1', 'Enzyme2', 'Percent_Similarity', 'E2_Sub', 'Neighbor_Substrate', 'Substrate_Distance', 'hit', 'Normalized_Distance']
    RANKER_FEATURES = ['Normalized_Distance', 'Percent_Similarity']
    ENZYME_COLUMN = 'Enzyme = 77'
    SUBSTRATE_COLUMN = 'Substrate = 17'

    def __init__(self, config=None):
        self.config = config or MSAConfig()
        self.interactions_df = None
        self.train_data = None
        self.test_data = None
        self.best_model = None
        self.total_substrates = None

    def _get_reaction_table(self):
        reaction_table = pd.read_csv(self.config.reaction_table)
        if self.ENZYME_COLUMN not in reaction_table.columns or self.SUBSTRATE_COLUMN not in reaction_table.columns:
            column_mapping = {}
            for col in reaction_table.columns:
                col_lower = col.lower()
                if 'enzyme' in col_lower and self.ENZYME_COLUMN not in column_mapping.values():
                    column_mapping[col] = self.ENZYME_COLUMN
                elif 'substrate' in col_lower and self.SUBSTRATE_COLUMN not in column_mapping.values():
                    column_mapping[col] = self.SUBSTRATE_COLUMN
            if column_mapping:
                reaction_table = reaction_table.rename(columns=column_mapping)
        if self.ENZYME_COLUMN not in reaction_table.columns or self.SUBSTRATE_COLUMN not in reaction_table.columns:
            raise ValueError(f'Cannot find required columns. Expected: {self.ENZYME_COLUMN}, {self.SUBSTRATE_COLUMN}. Actual columns: {list(reaction_table.columns)}')
        return reaction_table

    def _select_enzyme_data(self, dataset, enzyme_id):
        enzyme_data = dataset[dataset['Enzyme1'] == enzyme_id].copy()
        if enzyme_data.empty:
            return None
        missing_columns = [col for col in self.ESSENTIAL_COLUMNS if col not in enzyme_data.columns]
        if missing_columns:
            logger.warning(f'Enzyme {enzyme_id} missing columns: {missing_columns}, skipping')
            return None
        return enzyme_data[self.ESSENTIAL_COLUMNS].copy()

    def _apply_ranker(self, enzyme_data, model, deduplicate=True):
        ranked_data = enzyme_data.copy()
        features = ranked_data[self.RANKER_FEATURES].values
        ranked_data['ranker_score'] = model.predict(features)
        
        # 添加稳定的排序键：当ranker_score相同时，按原始索引排序确保跨平台一致性
        ranked_data['_sort_key'] = ranked_data.index
        ranked_data = ranked_data.sort_values(
            ['ranker_score', '_sort_key'], 
            ascending=[False, True]
        ).reset_index(drop=True)
        ranked_data = ranked_data.drop(columns=['_sort_key'])
        
        if deduplicate:
            ranked_data = ranked_data.drop_duplicates(subset='Neighbor_Substrate').reset_index(drop=True)
        return ranked_data

    def _count_total_examples(self, enzyme_id):
        return (self.interactions_df[self.ENZYME_COLUMN] == enzyme_id).sum()

    def prepare_data(self):
        logger.info('Starting data preprocessing...')
        logger.info('  - Loading reported sequences')
        train_seqs = list(SeqIO.parse(self.config.reported_sequences, 'fasta'))
        test_seqs = list(SeqIO.parse(self.config.ef_sequences, 'fasta'))
        logger.info(f'    Training set: {len(train_seqs)} sequences')
        logger.info(f'    Ef set: {len(test_seqs)} sequences')
        logger.info('    Using original sequence IDs, no renumbering')
        logger.info('  - Running MSA sequence alignment')
        self._run_msa_analysis(train_seqs, test_seqs)
        logger.info('  - Extracting top similar enzymes')
        self._extract_top_matches()
        logger.info('  - Getting neighbor substrates')
        self._get_neighbor_substrates()
        logger.info('  - Calculating substrate similarity')
        self._calculate_substrate_similarity()
        logger.info('  - Merging data and computing scores')
        self._merge_and_score_data()
        self.interactions_df = self._get_reaction_table()
        self.total_substrates = self.config.total_substrates
        logger.info(f'  - Total substrates: {self.total_substrates}')
        logger.info('Data preprocessing completed')

    def _run_msa_analysis(self, train_seqs, test_seqs):
        train_input = f'{self.config.prep_dir}/train_input.fasta'
        train_output = f'{self.config.prep_dir}/train_aligned.fasta'
        train_csv = f'{self.config.user_dir}/train_similarity.csv'
        SeqIO.write(train_seqs, train_input, 'fasta')
        cmd = [self.config.clustal_exe, '-i', train_input, '-o', train_output, '--outfmt=clu', '--force']
        subprocess.run(cmd, check=True)
        self._parse_alignment_results(train_output, train_csv)
        test_input = test_output = None
        if test_seqs:
            test_input = f'{self.config.prep_dir}/test_input.fasta'
            test_output = f'{self.config.prep_dir}/test_aligned.fasta'
            test_csv = f'{self.config.user_dir}/test_similarity.csv'
            SeqIO.write(test_seqs, test_input, 'fasta')
            test_cmd = [self.config.clustal_exe, '-i', test_input, '-o', test_output, '--outfmt=clu', '--force']
            subprocess.run(test_cmd, check=True)
            self._parse_alignment_results(test_output, test_csv)
            self._filter_test_similarity()
        for temp_file in [train_input, train_output, test_input, test_output]:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    def _parse_alignment_results(self, alignment_file, output_csv):
        sequences = list(SeqIO.parse(alignment_file, 'clustal'))
        similarity_matrix = []
        for (i, seq1) in enumerate(sequences):
            for (j, seq2) in enumerate(sequences):
                if i != j:
                    similarity = self._calculate_sequence_similarity(seq1.seq, seq2.seq)
                    similarity_matrix.append([seq1.id, seq2.id, similarity])
        df = pd.DataFrame(similarity_matrix, columns=['Enzyme1', 'Enzyme2', 'Percent_Similarity'])
        df.to_csv(output_csv, index=False)

    def _filter_test_similarity(self):
        test_similarity_file = f'{self.config.user_dir}/test_similarity.csv'
        if not os.path.exists(test_similarity_file):
            return
        enzyme_unique_df = pd.read_csv(self.config.reported_id)
        unique_enzyme_ids = set(enzyme_unique_df['Enzyme ID'].astype(str))
        test_df = pd.read_csv(test_similarity_file)
        filtered_df = test_df[(~test_df['Enzyme1'].astype(str).isin(unique_enzyme_ids)) & (test_df['Enzyme2'].astype(str).isin(unique_enzyme_ids))]
        filtered_df.to_csv(test_similarity_file, index=False)
        logger.info(f'   Test similarity filtered: {len(test_df)} -> {len(filtered_df)} records')
        if len(filtered_df) > 0:
            logger.info(f"    - Enzyme1 range: {filtered_df['Enzyme1'].min()}-{filtered_df['Enzyme1'].max()}")
            logger.info(f"    - Enzyme2 range: {filtered_df['Enzyme2'].min()}-{filtered_df['Enzyme2'].max()}")
            enzyme1_set = set(filtered_df['Enzyme1'].unique())
            enzyme2_set = set(filtered_df['Enzyme2'].unique())
            logger.info(f'    - Enzyme1 unique count: {len(enzyme1_set)}')
            logger.info(f'    - Enzyme2 unique count: {len(enzyme2_set)}')
            invalid_enzyme2 = enzyme2_set - unique_enzyme_ids
            if invalid_enzyme2:
                logger.info(f'     Enzyme2 values not in training set: {sorted(list(invalid_enzyme2))[:5]}...')
            invalid_enzyme1 = enzyme1_set & unique_enzyme_ids
            if invalid_enzyme1:
                logger.info(f'     Enzyme1 values in training set: {sorted(list(invalid_enzyme1))[:5]}...')
        else:
            logger.warning(f'     Filtered data is empty!')

    def _calculate_sequence_similarity(self, seq1, seq2):
        matches = sum((a == b for (a, b) in zip(str(seq1), str(seq2))))
        total_length = min(len(seq1), len(seq2))
        return matches / total_length * 100 if total_length > 0 else 0

    def _extract_top_matches(self):
        for dataset in ['train', 'test']:
            input_file = f'{self.config.user_dir}/{dataset}_similarity.csv'
            output_file = f'{self.config.user_dir}/{dataset}_top10.csv'
            if not os.path.exists(input_file):
                continue
            df = pd.read_csv(input_file)
            top_matches = []
            for enzyme_id in sorted(df['Enzyme1'].unique()):
                enzyme_data = df[df['Enzyme1'] == enzyme_id]
                top_10 = enzyme_data.nlargest(self.config.top_k, 'Percent_Similarity')
                top_matches.append(top_10)
            if top_matches:
                result_df = pd.concat(top_matches, ignore_index=True)
                result_df.to_csv(output_file, index=False)

    def _get_neighbor_substrates(self):
        reaction_table = self._get_reaction_table()
        for dataset in ['train', 'test']:
            input_file = f'{self.config.user_dir}/{dataset}_top10.csv'
            output_file = f'{self.config.user_dir}/{dataset}_top10_E2Sub.csv'
            if not os.path.exists(input_file):
                continue
            top_matches = pd.read_csv(input_file)
            expanded_data = []
            for idx in sorted(top_matches.index):
                row = top_matches.loc[idx]
                enzyme2_id = row['Enzyme2']
                substrates = reaction_table[reaction_table[self.ENZYME_COLUMN] == enzyme2_id][self.SUBSTRATE_COLUMN]
                for substrate in substrates:
                    new_row = row.copy()
                    new_row['E2_Sub'] = substrate
                    expanded_data.append(new_row)
            if expanded_data:
                result_df = pd.DataFrame(expanded_data)
                result_df.to_csv(output_file, index=False)

    def _calculate_substrate_similarity(self, n_components=None):
        if n_components is None:
            n_components = self.config.pca_components
        substrates_pcs = pd.read_csv(self.config.substrates_pcs)
        feature_columns = [col for col in substrates_pcs.columns if col.startswith('PC_')]
        feature_columns = feature_columns[:n_components]
        logger.info(f'    - Using PCA features: {feature_columns}')
        features = substrates_pcs[feature_columns].values
        tree = cKDTree(features)
        (distances, indices) = tree.query(features, k=self.config.num_neighbors)
        expanded_data = []
        for (i, (dist_list, idx_list)) in enumerate(zip(distances, indices)):
            substrate_id = substrates_pcs.iloc[i, 0]
            for (j, (dist, idx)) in enumerate(zip(dist_list[1:self.config.target_neighbors + 1], idx_list[1:self.config.target_neighbors + 1]), 1):
                neighbor_id = substrates_pcs.iloc[idx, 0]
                expanded_data.append([substrate_id, j, neighbor_id, dist])
        result_df = pd.DataFrame(expanded_data, columns=['Substrate', 'Rank', 'Neighbor_Substrate', 'Distance'])
        result_df.to_csv(f'{self.config.user_dir}/substrate_neighbors_{n_components}pcs.csv', index=False)

    def _merge_and_score_data(self):
        neighbors_path = f'{self.config.user_dir}/substrate_neighbors_{self.config.pca_components}pcs.csv'
        substrate_neighbors = pd.read_csv(neighbors_path)
        reaction_table = self._get_reaction_table()
        for dataset in ['train', 'test']:
            input_file = f'{self.config.user_dir}/{dataset}_top10_E2Sub.csv'
            output_file = f'{self.config.user_dir}/{dataset}_final_data.csv'
            if not os.path.exists(input_file):
                continue
            data = pd.read_csv(input_file)
            expanded_data = []
            for idx in sorted(data.index):
                row = data.loc[idx]
                e2_sub = row['E2_Sub']
                neighbors = substrate_neighbors[substrate_neighbors['Substrate'] == e2_sub]
                for neighbor_idx in sorted(neighbors.index):
                    neighbor = neighbors.loc[neighbor_idx]
                    new_row = row.copy()
                    new_row['Neighbor_Substrate'] = neighbor['Neighbor_Substrate']
                    new_row['Substrate_Distance'] = neighbor['Distance']
                    has_reaction = ((reaction_table[self.ENZYME_COLUMN] == row['Enzyme1']) & (reaction_table[self.SUBSTRATE_COLUMN] == neighbor['Neighbor_Substrate'])).any()
                    new_row['hit'] = int(has_reaction)
                    expanded_data.append(new_row)
            if expanded_data:
                result_df = pd.DataFrame(expanded_data)
                max_distance = result_df['Substrate_Distance'].max()
                if max_distance > 0:
                    result_df['Normalized_Distance'] = result_df['Substrate_Distance'] / max_distance
                else:
                    result_df['Normalized_Distance'] = 0.0
                result_df['MSA_Score'] = np.sqrt(result_df['Normalized_Distance'] ** 2 + (1 - result_df['Percent_Similarity'] / 100) ** 2)
                result_df.to_csv(output_file, index=False)
                if dataset == 'train':
                    self.train_data = result_df
                    logger.info(f"   Training data: {len(result_df)} records, positive samples: {result_df['hit'].sum()}")
                else:
                    self.test_data = result_df
                    logger.info(f"   Test data: {len(result_df)} records, positive samples: {result_df['hit'].sum()}")

    def train_models(self):
        logger.info('Starting model training...')
        if self.train_data is None:
            raise ValueError('Training data not prepared. Please call prepare_data() first.')
        X = self.train_data[['Normalized_Distance', 'Percent_Similarity']].values
        y = self.train_data['hit'].values
        group_id = self.train_data['Enzyme1'].astype(int).values
        logger.info(f'    - Training set: {len(X)} samples')
        logger.info(f'    - Positive samples: {sum(y)}, Negative samples: {len(y) - sum(y)}')
        best_model = None
        best_score = -np.inf
        best_params = {}
        grid_results = []
        param_grid = {'iterations': [50, 100, 200, 300, 400, 500, 600, 800, 1000], 'depth': [1, 2, 3, 4, 5, 6, 8, 10, 12]}
        total_combinations = len(param_grid['iterations']) * len(param_grid['depth'])
        logger.info(f'Starting grid search with {total_combinations} parameter combinations...')
        current_count = 0
        for iterations in param_grid['iterations']:
            for depth in param_grid['depth']:
                current_count += 1
                progress = current_count / total_combinations * 100
                logger.info(f'Progress: {current_count}/{total_combinations} ({progress:.1f}%) - iters={iterations}, depth={depth}')
                model = CatBoostRanker(iterations=iterations, depth=depth, loss_function='YetiRank', verbose=False, random_state=self.config.random_state)
                try:
                    train_pool = Pool(data=X, label=y, group_id=group_id)
                    model.fit(train_pool)
                    model_filename = f'iters{iterations}_depth{depth}.cbm'
                    model_path = f'{self.config.models_dir}/{model_filename}'
                    if os.path.exists(model_path):
                        logger.warning(f'  File exists, will be overwritten: {model_path}')
                    model.save_model(model_path)
                    if not os.path.exists(model_path):
                        logger.error(f'  Failed to save model: {model_filename}')
                        continue
                    logger.debug(f'  Saved model: {model_filename} ({os.path.getsize(model_path)} bytes)')
                    score = self._evaluate_model_on_train(model)
                    logger.info(f'   Training set NDCG@20: {score:.4f} → {model_filename}')
                    grid_results.append({'iterations': iterations, 'depth': depth, 'ndcg@20': score, 'model_file': model_filename})
                except Exception as e:
                    logger.error(f' Training/saving model failed: iters={iterations}, depth={depth}')
                    logger.error(f'   Error: {e}')
                    grid_results.append({'iterations': iterations, 'depth': depth, 'ndcg@20': np.nan, 'model_file': f'iters{iterations}_depth{depth}.cbm', 'error': str(e)})
                    continue
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = {'iterations': iterations, 'depth': depth}
                    logger.info(f' Found better model! NDCG@20: {score:.4f}')
        if not grid_results:
            raise RuntimeError('No models were successfully trained. Please check your data and configuration.')
        grid_df = pd.DataFrame(grid_results)
        grid_df = grid_df.sort_values('ndcg@20', ascending=False, na_position='last')
        grid_output_path = f'{self.config.models_dir}/msa_grid_search_results.csv'
        grid_df.to_csv(grid_output_path, index=False)
        logger.info(f'Grid search results saved: {grid_output_path}')
        if best_model is None:
            raise RuntimeError('No valid model was found. All training attempts failed.')
        logger.info(f"  Best parameters: iterations={best_params['iterations']}, depth={best_params['depth']}, NDCG@20={best_score:.4f}")
        best_filename = f"iters{best_params['iterations']}_depth{best_params['depth']}_best_ranker.cbm"
        best_path = f'{self.config.models_dir}/{best_filename}'
        best_model.save_model(best_path)
        logger.info(f'Best model parameters: {best_params}')
        logger.info(f'Best NDCG@20 score: {best_score:.4f}')
        self.best_model = best_model
        return best_model

    def _evaluate_model_on_train(self, model):
        if self.train_data is None or len(self.train_data) == 0:
            logger.warning('     Training data is empty, returning -inf')
            return -np.inf
        metrics_list = []
        for enzyme_id in sorted(self.train_data['Enzyme1'].unique()):
            enzyme_data = self._select_enzyme_data(self.train_data, enzyme_id)
            if enzyme_data is None:
                continue
            ranked_data = self._apply_ranker(enzyme_data, model, deduplicate=False)
            hits = ranked_data['hit'].values
            metrics = self._calculate_metrics(hits)
            metrics_list.append(metrics)
        avg_metrics = self._average_metrics(metrics_list)
        return avg_metrics.get('ndcg@20', -np.inf)

    def _calculate_metrics(self, hits):
        hits = np.array(hits)
        if len(hits) == 0:
            return {'ndcg@20': 0.0}
        k = 20
        hits_at_k = hits[:k]
        if len(hits_at_k) == 0:
            return {'ndcg@20': 0.0}
        try:
            scores_at_k = np.arange(len(hits_at_k))[::-1]
            ndcg_at_20 = ndcg_score([hits_at_k], [scores_at_k])
        except Exception:
            ndcg_at_20 = 0.0
        return {'ndcg@20': ndcg_at_20}

    def run_complete_pipeline(self):
        logger.info('Starting complete MSA pipeline...')
        logger.info('=' * 60)
        try:
            self.prepare_data()
            self.train_models()
            self._generate_predictions()
            logger.info('Complete pipeline executed successfully!')
        except Exception as e:
            logger.error(f'Pipeline execution failed: {e}')
            raise

    def _average_metrics(self, all_metrics):
        if not all_metrics:
            return {'ndcg@20': 0.0}
        all_metrics_df = pd.DataFrame(all_metrics)
        avg_metrics = all_metrics_df.mean()
        return avg_metrics.to_dict()

    def _generate_predictions(self):
        if not hasattr(self, 'best_model') or self.best_model is None:
            logger.warning('  No trained model found, skipping prediction export')
            return
        if self.test_data is None or len(self.test_data) == 0:
            logger.warning('  Test data is empty, skipping prediction export')
            return
        logger.info('Generating prediction CSV file for Test data...')
        dataset_predictions = []
        for enzyme_id in sorted(self.test_data['Enzyme1'].unique()):
            enzyme_data = self._select_enzyme_data(self.test_data, enzyme_id)
            if enzyme_data is None:
                continue
            ranked_data = self._apply_ranker(enzyme_data, self.best_model, deduplicate=True)
            dataset_predictions.append(ranked_data)
        if not dataset_predictions:
            logger.warning('  No predictions generated for Test data.')
            return
        predictions = pd.concat(dataset_predictions, ignore_index=True)
        predictions.to_csv(f'{self.config.models_dir}/predictions.csv', index=False)
        logger.info('  Saved predictions.csv')

    def _load_best_model_from_disk(self):
        models_dir = self.config.models_dir
        candidate_files = [f for f in os.listdir(models_dir) if f.endswith('_best_ranker.cbm')]
        if not candidate_files:
            raise FileNotFoundError(f'No *_best_ranker.cbm model file found in {models_dir}. Please run hyperparameter search first.')
        candidate_paths = [os.path.join(models_dir, f) for f in candidate_files]
        best_path = max(candidate_paths, key=os.path.getmtime)
        logger.info(f'Loading best model from: {best_path}')
        model = CatBoostRanker()
        model.load_model(best_path)
        self.best_model = model
        return model

def main():
    logger.info('MSA Prediction System')
    logger.info('=' * 60)
    pipeline = MSAPipeline()
    pipeline.run_complete_pipeline()
    return pipeline


def run_grid_search():
    logger.info('=== Running hyperparameter search (data prep + training) ===')
    pipeline = MSAPipeline()
    pipeline.prepare_data()
    pipeline.train_models()
    logger.info('=== Grid search finished ===')
    return pipeline


def run_prediction_only():
    logger.info('=== Running prediction generation using best saved model ===')
    pipeline = MSAPipeline()
    pipeline.prepare_data()
    pipeline._load_best_model_from_disk()
    pipeline._generate_predictions()
    logger.info('=== Prediction finished ===')
    return pipeline
if __name__ == '__main__':
    main()
