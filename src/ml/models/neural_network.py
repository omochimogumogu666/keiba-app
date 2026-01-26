"""
Neural Network model for horse racing prediction.

PyTorchベースのニューラルネットワークモデル。
BaseRaceModelインターフェースを実装し、既存パイプラインと統合。
"""
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.ml.models.base_model import BaseRaceModel
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class RaceNet(nn.Module):
    """
    競馬予測用ニューラルネットワークアーキテクチャ。

    Input(n_features) → Dense(256) → Dense(128) → Dense(64) → Output
    各層にBatchNorm + ReLU + Dropoutを適用
    """

    def __init__(
        self,
        n_features: int,
        task: str = 'regression',
        hidden_sizes: List[int] = None,
        dropout_rate: float = 0.3,
        n_classes: int = 2
    ):
        """
        ネットワークを初期化。

        Args:
            n_features: 入力特徴量の数
            task: 'regression' or 'classification'
            hidden_sizes: 隠れ層のサイズリスト
            dropout_rate: ドロップアウト率
            n_classes: 分類タスクのクラス数
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.task = task

        # Build layers
        layers = []
        in_features = n_features

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        if task == 'regression':
            self.output_layer = nn.Linear(in_features, 1)
        else:
            self.output_layer = nn.Linear(in_features, n_classes)

    def forward(self, x):
        """フォワードパス"""
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        if self.task == 'regression':
            return x.squeeze(-1)
        return x


class NeuralNetworkRaceModel(BaseRaceModel):
    """
    PyTorchベースの競馬予測モデル。

    回帰（着順予測）と分類（勝敗予測）の両タスクをサポート。
    """

    def __init__(
        self,
        task: str = 'regression',
        hidden_sizes: List[int] = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        n_epochs: int = 100,
        early_stopping_patience: int = 10,
        random_state: int = 42,
        version: str = "1.0",
        device: str = None,
        **kwargs
    ):
        """
        ニューラルネットワークモデルを初期化。

        Args:
            task: 'regression' or 'classification'
            hidden_sizes: 隠れ層のサイズリスト
            dropout_rate: ドロップアウト率
            learning_rate: 学習率
            batch_size: バッチサイズ
            n_epochs: 最大エポック数
            early_stopping_patience: Early stoppingの許容エポック数
            random_state: 乱数シード
            version: モデルバージョン
            device: 'cuda' or 'cpu' (None = auto detect)
            **kwargs: 追加パラメータ
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        super().__init__(model_name=f'NeuralNetwork_{task}', version=version)

        self.task = task
        self.hidden_sizes = hidden_sizes or [256, 128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        # デバイス設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # パラメータ保存
        self.params = {
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'early_stopping_patience': early_stopping_patience,
            'random_state': random_state,
            **kwargs
        }

        # モデルとスケーラーは学習時に初期化
        self.model = None
        self.scaler = StandardScaler()
        self.n_classes = 2  # 分類タスク用

        # 乱数シード設定
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        shuffle: bool = True
    ) -> DataLoader:
        """DataLoaderを作成"""
        X_tensor = torch.FloatTensor(X)

        if y is not None:
            if self.task == 'classification':
                y_tensor = torch.LongTensor(y.astype(int))
            else:
                y_tensor = torch.FloatTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True,
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        ニューラルネットワークを学習。

        Args:
            X_train: 学習特徴量
            y_train: 学習ラベル
            X_val: 検証特徴量 (optional)
            y_val: 検証ラベル (optional)
            verbose: 進捗を表示するか
            progress_callback: 進捗を報告するコールバック関数
            cancel_check: キャンセル確認用コールバック関数
            **kwargs: 追加パラメータ

        Returns:
            学習メトリクスの辞書
        """
        logger.info(f"Training {self.model_name} on {len(X_train)} samples (device: {self.device})")

        # 特徴量カラムを保存
        self.feature_columns = X_train.columns.tolist()

        # データをスケーリング
        X_train_scaled = self.scaler.fit_transform(X_train.values)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val.values)

        # 分類タスクの場合、クラス数を確認
        if self.task == 'classification':
            self.n_classes = len(np.unique(y_train))

        # モデルを初期化
        n_features = X_train.shape[1]
        self.model = RaceNet(
            n_features=n_features,
            task=self.task,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate,
            n_classes=self.n_classes
        ).to(self.device)

        # DataLoaderを作成
        train_loader = self._create_dataloader(X_train_scaled, y_train.values, shuffle=True)

        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val_scaled, y_val.values, shuffle=False)
        else:
            val_loader = None

        # 損失関数とオプティマイザ
        if self.task == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 学習ループ
        train_start = datetime.utcnow()
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        training_history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.n_epochs):
            # キャンセルチェック
            if cancel_check and cancel_check():
                logger.info(f"Training cancelled at epoch {epoch}")
                break

            # 学習フェーズ
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            train_loss /= len(train_loader.dataset)
            training_history['train_loss'].append(train_loss)

            # 検証フェーズ
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)

                val_loss /= len(val_loader.dataset)
                training_history['val_loss'].append(val_loss)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.n_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    # 進捗コールバック（early stopping）
                    if progress_callback:
                        progress_callback({
                            'event': 'epoch_complete',
                            'epoch': epoch + 1,
                            'total_epochs': self.n_epochs,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'early_stopped': True
                        })
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.4f}")

            # 進捗コールバック
            if progress_callback:
                progress_callback({
                    'event': 'epoch_complete',
                    'epoch': epoch + 1,
                    'total_epochs': self.n_epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

        # ベストモデルを復元
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        train_time = (datetime.utcnow() - train_start).total_seconds()
        self.is_trained = True

        # 評価メトリクスを計算
        metrics = self._calculate_metrics(
            X_train_scaled, y_train.values,
            X_val_scaled if X_val is not None else None,
            y_val.values if y_val is not None else None,
            train_time,
            len(training_history['train_loss'])
        )

        # メタデータを保存
        self.training_metadata = {
            'trained_at': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'params': self.params,
            'training_history': training_history
        }

        return metrics

    def _calculate_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        train_time: float,
        n_epochs_trained: int
    ) -> Dict[str, Any]:
        """評価メトリクスを計算"""
        self.model.eval()

        metrics = {
            'training_time_seconds': train_time,
            'n_samples_train': len(X_train),
            'n_features': len(self.feature_columns),
            'task': self.task,
            'n_epochs_trained': n_epochs_trained,
            'device': str(self.device)
        }

        # 学習データでの予測
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            train_pred = self.model(X_tensor).cpu().numpy()

        if self.task == 'regression':
            train_mse = mean_squared_error(y_train, train_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            train_rmse = np.sqrt(train_mse)

            metrics.update({
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_rmse': train_rmse
            })

            logger.info(f"Training RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")

        else:
            train_pred_class = train_pred.argmax(axis=1)
            train_acc = accuracy_score(y_train, train_pred_class)
            metrics['train_accuracy'] = train_acc

            logger.info(f"Training accuracy: {train_acc:.4f}")

        # 検証データでの予測
        if X_val is not None and y_val is not None:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_val).to(self.device)
                val_pred = self.model(X_tensor).cpu().numpy()

            metrics['n_samples_val'] = len(X_val)

            if self.task == 'regression':
                val_mse = mean_squared_error(y_val, val_pred)
                val_mae = mean_absolute_error(y_val, val_pred)
                val_rmse = np.sqrt(val_mse)

                metrics.update({
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse
                })

                logger.info(f"Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

            else:
                val_pred_class = val_pred.argmax(axis=1)
                val_acc = accuracy_score(y_val, val_pred_class)
                metrics['val_accuracy'] = val_acc

                logger.info(f"Validation accuracy: {val_acc:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行。

        Args:
            X: 特徴量DataFrame

        Returns:
            予測値の配列
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_prepared = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared.values)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        if self.task == 'classification':
            return predictions.argmax(axis=1)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        クラス確率を予測（分類タスク用）。

        Args:
            X: 特徴量DataFrame

        Returns:
            クラス確率の配列
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification task")

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_prepared = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared.values)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            logits = self.model(X_tensor)
            probas = torch.softmax(logits, dim=1).cpu().numpy()

        return probas

    def predict_top_n(
        self,
        X: pd.DataFrame,
        n: int = 3,
        return_probabilities: bool = False
    ) -> pd.DataFrame:
        """
        各レースでトップNの馬を予測。

        Args:
            X: 特徴量DataFrame（race_id, horse_id カラムを含む）
            n: 各レースで返す馬の数
            return_probabilities: 確率を含めるか（分類タスクのみ）

        Returns:
            race_id, horse_id, rank を含むDataFrame
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if 'race_id' not in X.columns or 'horse_id' not in X.columns:
            raise ValueError("X must include 'race_id' and 'horse_id' columns")

        if self.task == 'regression':
            predictions = self.predict(X)
            X_with_pred = X[['race_id', 'horse_id']].copy()
            X_with_pred['predicted_position'] = predictions

            # 予測着順でランク付け（低いほど良い）
            X_with_pred['rank'] = X_with_pred.groupby('race_id')['predicted_position'].rank(method='first')

        else:
            probas = self.predict_proba(X)
            win_proba = probas[:, 1] if probas.shape[1] == 2 else probas[:, -1]

            X_with_pred = X[['race_id', 'horse_id']].copy()
            X_with_pred['win_probability'] = win_proba

            # 勝利確率でランク付け（高いほど良い）
            X_with_pred['rank'] = X_with_pred.groupby('race_id')['win_probability'].rank(
                method='first', ascending=False
            )

        # トップNをフィルタ
        top_n = X_with_pred[X_with_pred['rank'] <= n].copy()
        top_n = top_n.sort_values(['race_id', 'rank'])

        return top_n

    def get_feature_importance(self) -> pd.Series:
        """
        特徴量重要度を取得（順列重要度による近似）。

        Note: NNは直接的な特徴量重要度を持たないため、
              この実装では入力層の重みの絶対値平均を使用。

        Returns:
            特徴量名をインデックス、重要度スコアを値とするSeries
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        # 入力層の重みを取得
        first_layer = self.model.hidden_layers[0]
        if isinstance(first_layer, nn.Linear):
            weights = first_layer.weight.data.cpu().numpy()
            importance = np.abs(weights).mean(axis=0)

            return pd.Series(
                importance,
                index=self.feature_columns,
                name='importance'
            ).sort_values(ascending=False)

        raise NotImplementedError("Could not extract feature importance")

    def save(self, filepath: str) -> None:
        """
        モデルを.pt形式で保存。

        Args:
            filepath: 保存パス（.ptを推奨）
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 拡張子を.ptに変更
        if not filepath.endswith('.pt'):
            filepath = filepath.rsplit('.', 1)[0] + '.pt'

        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_features': len(self.feature_columns),
                'task': self.task,
                'hidden_sizes': self.hidden_sizes,
                'dropout_rate': self.dropout_rate,
                'n_classes': self.n_classes
            },
            'model_name': self.model_name,
            'version': self.version,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'task': self.task,
            'params': self.params,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }

        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        モデルを読み込み。

        Args:
            filepath: 読み込みパス
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = torch.load(filepath, map_location=self.device)

        # モデル設定を復元
        config = model_data['model_config']
        self.task = config['task']
        self.hidden_sizes = config['hidden_sizes']
        self.dropout_rate = config['dropout_rate']
        self.n_classes = config['n_classes']

        # モデルを再構築
        self.model = RaceNet(
            n_features=config['n_features'],
            task=self.task,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate,
            n_classes=self.n_classes
        ).to(self.device)

        self.model.load_state_dict(model_data['model_state_dict'])

        # その他の属性を復元
        self.model_name = model_data['model_name']
        self.version = model_data['version']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        self.training_metadata = model_data.get('training_metadata', {})
        self.params = model_data.get('params', {})

        # スケーラーを復元
        self.scaler.mean_ = model_data['scaler_mean']
        self.scaler.scale_ = model_data['scaler_scale']

        self.model.eval()
        logger.info(f"Model loaded from {filepath} (task: {self.task})")
