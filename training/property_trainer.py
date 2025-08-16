"""分子性质预测训练器

该模块实现分子性质预测模型的训练流程。
主要功能：
1. 下游任务数据加载和处理
2. 模型微调训练
3. 评估和验证
4. 早停和模型保存
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
from symbol_dictionary import SPECIAL_TOKENS
from models.property_model import PropertyPredictionModel, ImbalancedBCELoss,PropertyLoss, calculate_metrics, EarlyStopping, PropertyPredConfig
import pandas as pd
from torch.utils.data import Dataset


class PropertyDataset(Dataset):
    """分子性质预测数据集"""

    def __init__(self, 
                 smiles_ids: List[List[int]],
                 labels: List[List[float]]):
        """初始化分子性质预测数据集

        Args:
            smiles_ids: SMILES的ID序列列表
            labels: 标签列表
        """
        self.smiles_ids = smiles_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.smiles_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles_ids = self.smiles_ids[idx]
        label = self.labels[idx]

        return {
            'smiles': smiles_ids,  # 返回列表，稍后在collate中处理
            'labels': torch.tensor(label, dtype=torch.float)
        }



def collate_property_batch(batch):
    """分子性质预测数据的自定义collate函数"""
    # 获取最大长度
    max_len = max(len(item['smiles']) for item in batch)

    # 填充序列
    smiles_batch = []
    labels_batch = []

    for item in batch:
        # 填充SMILES序列
        smiles = item['smiles'] + [SPECIAL_TOKENS.PAD_ID] * \
            (max_len - len(item['smiles']))
        smiles_batch.append(smiles)
        labels_batch.append(item['labels'])

    return {
        'smiles': torch.tensor(smiles_batch, dtype=torch.long),
        'labels': torch.stack(labels_batch)
    }

class PropertyTrainer:
    """分子性质预测训练器

    负责分子性质预测模型的完整训练和评估流程。
    """

    def __init__(self, 
                 config: PropertyPredConfig,
                 save_dir: str, 
                 device: str = 'cpu'):
        """初始化分子性质预测训练器

        Args:
            config: 模型配置
            data_config: 数据配置
            save_dir: 模型保存目录
            device: 训练设备
        """
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)


        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None

        # 训练状态
        self.current_epoch = 0
        self.best_score = None
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def setup_model(self, final_pos_weights):
        """设置模型

        Args:
            num_tokens: 词汇表大小
        """
        print(f"初始化分子性质预测模型...")
        self.model = PropertyPredictionModel(
            self.config
        )
        self.model.to(self.device)

        # 设置损失函数
        # self.criterion = PropertyLoss()
        self.criterion = ImbalancedBCELoss(final_pos_weights)

        # 设置早停
        monitor_mode = 'max'
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode=monitor_mode
        )

    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 分别设置不同部分的学习率
        pretrained_params = []
        new_params = []

        for name, param in self.model.named_parameters():
            if 'transformer' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)

        self.optimizer = AdamW([
            {'params': pretrained_params, 'lr': self.config.learning_rate * 0.1},
            {'params': new_params, 'lr': self.config.learning_rate}
        ], weight_decay=0.01)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def load_data(self, csv_path: str) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
        """加载数据
        Returns:
            Tuple: (训练集, 验证集, 测试集, 任务信息)
        """
        print(f"加载数据: {csv_path}")


        # 创建数据集
        df = pd.read_csv(csv_path)
        smiles_ids = df.token_ids.values.tolist()
        labels = df.label_ids.values.tolist()
        smiles_ids = [eval(smiles_id) for smiles_id in smiles_ids]
        labels = [eval(label) for label in labels]
        dataset = PropertyDataset(smiles_ids, labels)
        
        # 划分数据集
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        
        # 计算每个类别的正负样本比例
        y_train_cb = torch.stack([dataset[i]['labels'] for i in train_dataset.indices])
        y_train_sum = y_train_cb.sum(axis=0)
        total_samples = len(y_train_cb)


        # 使用有效样本数计算权重 (Effective Number of Samples)
        def calculate_effective_weights(y_train, beta=0.9999):
            """
            计算有效样本数权重，处理长尾分布
            """
            pos_counts = y_train.sum(axis=0)
            neg_counts = len(y_train) - pos_counts
            
            # 有效样本数公式
            effective_pos = (1 - beta ** pos_counts) / (1 - beta)
            effective_neg = (1 - beta ** neg_counts) / (1 - beta)
            
            # 计算权重
            weights = effective_neg / effective_pos
            
            # 归一化权重
            weights = weights / weights.mean()
            
            return weights

        # 计算多种权重策略
        # 策略1: 传统逆频率权重
        traditional_weights = (total_samples - y_train_sum) / y_train_sum

        # 策略2: 有效样本数权重
        effective_weights = calculate_effective_weights(y_train_cb)

        # 策略3: 平衡权重 (类似sklearn的balanced)
        balanced_weights = total_samples / (2 * y_train_sum)

        # 策略4: 组合权重 (传统 + 有效样本数)
        combined_weights = 0.6 * traditional_weights + 0.4 * effective_weights

        print(f"类别权重统计:")
        print(f"  传统权重 - 均值: {traditional_weights.mean():.3f}, 标准差: {traditional_weights.std():.3f}")
        print(f"  有效权重 - 均值: {effective_weights.mean():.3f}, 标准差: {effective_weights.std():.3f}")
        print(f"  平衡权重 - 均值: {balanced_weights.mean():.3f}, 标准差: {balanced_weights.std():.3f}")
        print(f"  组合权重 - 均值: {combined_weights.mean():.3f}, 标准差: {combined_weights.std():.3f}")

        # 选择组合权重策略
        final_pos_weights = torch.FloatTensor(combined_weights).to("cpu")

        print(f"\n选择组合权重策略，权重范围: [{final_pos_weights.min():.3f}, {final_pos_weights.max():.3f}]")
                
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
            collate_fn=collate_property_batch
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=collate_property_batch
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=collate_property_batch
        )

        print(f"数据划分:")
        print(f"  训练集: {len(train_dataset)}")
        print(f"  验证集: {len(val_dataset)}")
        print(f"  测试集: {len(test_dataset)}")

        return train_loader, val_loader, test_loader, final_pos_weights

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch

        Args:
            dataloader: 数据加载器

        Returns:
            Dict[str, float]: 训练指标
        """
        self.model.train()

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # 数据移到设备
            smiles_ids = batch['smiles'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            predictions = self.model(smiles_ids)

            # 计算损失
            loss = self.criterion(predictions, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 累计指标
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(labels.detach().cpu())
            num_batches += 1

            # 打印进度
            if batch_idx % 50 == 0:
                print(
                    f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # 计算整体指标
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            metrics = calculate_metrics(
                all_predictions, all_targets,
            )
            metrics['loss'] = total_loss / num_batches
        else:
            # 如果没有数据，返回默认指标
            metrics = {'loss': 0.0, 'accuracy': 0.0}

        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型

        Args:
            dataloader: 数据加载器

        Returns:
            Dict[str, float]: 评估指标
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # 数据移到设备
                smiles_ids = batch['smiles'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                predictions = self.model(smiles_ids)

                # 计算损失
                loss = self.criterion(predictions, labels)

                # 累计指标
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
                num_batches += 1

        # 计算整体指标
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            metrics = calculate_metrics(
                all_predictions, all_targets,
            )
            metrics['loss'] = total_loss / num_batches

            # 计算额外指标
            if all_predictions.size(1) == 1:
                # 二分类 - 计算AUC
                try:
                    auc = roc_auc_score(
                        all_targets.numpy(),
                        all_predictions.numpy()
                    )
                    metrics['auc'] = auc
                except:
                    pass
       
        else:
            # 如果没有数据，返回默认指标
            metrics = {'loss': 0.0, 'accuracy': 0.0}

        return metrics

    def save_checkpoint(self, epoch: int, train_metrics: Dict[str, float],
                        val_metrics: Dict[str, float], is_best: bool = False):
        """保存检查点

        Args:
            epoch: 当前epoch
            train_metrics: 训练指标
            val_metrics: 验证指标
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': self.train_metrics,
            'val_metrics_history': self.val_metrics
        }

        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)

        # 如果是最佳模型，额外保存
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  保存最佳模型")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点

        Args:
            checkpoint_path: 检查点路径

        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(checkpoint_path):
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint['epoch']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_metrics = checkpoint.get('train_metrics_history', [])
            self.val_metrics = checkpoint.get('val_metrics_history', [])

            print(f"成功加载检查点 (epoch {self.current_epoch})")
            return True

        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False

    def train(self, 
              csv_path: str,
              pretrain_model_path: str = None,
              resume_from_checkpoint: bool = False):
        """开始训练

        Args:
            pretrain_model_path: 预训练模型路径
            resume_from_checkpoint: 是否从检查点恢复训练
        """
        print(f"开始训练分子性质预测模型")

        # 加载数据
        train_loader, val_loader, test_loader, final_pos_weights = self.load_data(csv_path)

        self.setup_model(final_pos_weights)

        # 加载预训练权重
        if pretrain_model_path:
            print(f"加载预训练模型: {pretrain_model_path}")
            self.model.load_pretrained_encoder(
                pretrain_model_path,
                freeze_encoder=self.config.freeze_encoder,
                unfreeze_last_layers=self.config.unfreeze_last_layers
            )
            # 打印模型信息
        model_info = self.model.get_model_info()
        print(f"模型信息:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        # 设置优化器
        self.setup_optimizer()

        # 尝试恢复训练
        if resume_from_checkpoint:
            checkpoint_path = os.path.join(
                self.save_dir, 'latest_checkpoint.pt')
            self.load_checkpoint(checkpoint_path)

        # 训练循环
        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epoch):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epoch}")
            print("-" * 50)

            # 训练
            epoch_start = time.time()
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.evaluate(val_loader)
            epoch_time = time.time() - epoch_start

            # 记录指标
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)

            # 打印结果
            print(f"训练结果:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")

            print(f"  Train Acc: {train_metrics.get('accuracy', 0):.4f}")
            print(f"  Val Acc: {val_metrics.get('accuracy', 0):.4f}")
            if 'auc' in val_metrics:
                print(f"  Val AUC: {val_metrics['auc']:.4f}")
            if 'f1_macro' in train_metrics:
                print(f"  Train F1-Macro: {train_metrics['f1_macro']:.4f}")
            if 'f1_macro' in val_metrics:
                print(f"  Val F1-Macro: {val_metrics['f1_macro']:.4f}")
            if 'auc_macro' in train_metrics:
                print(
                    f"  Train AUC-Macro: {train_metrics['auc_macro']:.4f}")
            if 'auc_macro' in val_metrics:
                print(f"  Val AUC-Macro: {val_metrics['auc_macro']:.4f}")

            print(f"  Time: {epoch_time:.2f}s")

            # 学习率调度
            self.scheduler.step(val_metrics['loss'])

            # 检查是否为最佳模型
            monitor_metric = val_metrics.get('accuracy', -val_metrics['loss'])
            is_best = (self.best_score is None or
                       monitor_metric > self.best_score)
            if is_best:
                self.best_score = monitor_metric

            # 保存检查点
            self.save_checkpoint(epoch + 1, train_metrics,
                                 val_metrics, is_best)

            # 早停检查
            if self.early_stopping(monitor_metric):
                print(f"早停触发，停止训练")
                break

            # 更新当前epoch
            self.current_epoch = epoch + 1

        total_time = time.time() - start_time
        print(f"\n训练完成! 总用时: {total_time:.2f}s")

        # 测试集评估
        print("\n在测试集上评估...")
        test_metrics = self.evaluate(test_loader)
        print(f"测试结果:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")

        return test_metrics

    def get_training_history(self) -> Dict[str, List]:
        """获取训练历史"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
