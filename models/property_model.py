"""分子性质预测模型

该模块实现基于预训练Transformer的分子性质预测模型。
主要功能：
1. 分类任务预测
2. 回归任务预测
3. 迁移学习支持
"""

from sklearn.metrics import f1_score, roc_auc_score  # 新增的导入
from typing import Dict
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from .transformer_model import BaseTransformerModel
from symbol_dictionary import SPECIAL_TOKENS
from dataclasses import dataclass
import numpy as np


@dataclass
class PropertyPredConfig:
    """分子性质预测模型配置"""
    # 模型架构参数
    dim_embed: int = 128          # 嵌入维度
    dim_tf_hidden: int = 512      # Transformer隐藏层维度
    num_head: int = 8             # 多头注意力头数
    num_layers: int = 10          # Transformer层数
    dropout: float = 0.1          # Dropout率
    norm_first: bool = True       # 是否先进行归一化
    activation: str = 'gelu'      # 激活函数

    # 训练参数
    learning_rate: float = 0.0001  # 学习率
    num_epoch: int = 300           # 训练轮数
    batch_size: int = 32          # 批次大小
    init_range: float = 0.1       # 参数初始化范围

    # 迁移学习参数
    pretrain_path: Optional[str] = None  # 预训练模型路径
    freeze_encoder: bool = True          # 是否冻结编码器
    fine_tune_last_layer: bool = True    # 是否微调最后一层
    unfreeze_last_layers: int = 2        # 解冻最后几层
    transfer_learning_rate_ratio: float = 1.0  # 迁移学习学习率比例

    # 训练控制参数
    early_stopping_patience: int = 100   # 早停耐心值

    # 数据参数
    limit_smiles_length: int = 100  # SMILES最大长度
    fold_cv: int = 5                # 交叉验证折数
    
    num_tokens: int = 306
    num_labels: int = 138

class PropertyPredictionModel(nn.Module):
    """分子性质预测模型

    基于预训练Transformer编码器的分子性质预测模型，
    支持分类和回归任务。
    """

    def __init__(self, 
                 config: PropertyPredConfig):
        """初始化分子性质预测模型

        Args:
            config: 模型配置
            num_tokens: 词汇表大小
            num_labels: 标签数量
        """
        super().__init__()

        self.config = config

        # 基础Transformer模型
        self.transformer = BaseTransformerModel(
            num_tokens=config.num_tokens,
            d_model=config.dim_embed,
            num_heads=config.num_head,
            d_hidden=config.dim_tf_hidden,
            num_layers=config.num_layers,
            dropout=config.dropout,
            norm_first=config.norm_first,
            activation=config.activation,
            init_range=config.init_range
        )

        # 分类/回归头
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_embed, config.dim_embed // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_embed // 2, config.num_labels)
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, smiles_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播

        Args:
            smiles_ids: SMILES ID序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            torch.Tensor: 预测结果 [batch_size, num_labels]
        """
        batch_size = smiles_ids.size(0)
        device = smiles_ids.device

        # 添加CLS标记
        cls_tokens = torch.ones(
            batch_size, 1, dtype=torch.long, device=device) * SPECIAL_TOKENS.CLS_ID
        x = torch.cat([cls_tokens, smiles_ids], dim=1)

        # 创建填充掩码
        if attention_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([cls_mask, attention_mask == 0], dim=1)
        else:
            padding_mask = (x == SPECIAL_TOKENS.PAD_ID)

        # Transformer编码
        encoded = self.transformer(x, padding_mask)

        # 使用CLS位置的表示进行分类/回归
        cls_representation = encoded[:, 0, :]  # [batch_size, d_model]

        # 分类/回归头
        logits = self.classifier(cls_representation)

        # 应用输出激活函数
        output = self.output_activation(logits)

        return output

    def load_pretrained_encoder(self, pretrain_model_path: str,
                                freeze_encoder: bool = True,
                                unfreeze_last_layers: int = 0):
        """加载预训练编码器

        Args:
            pretrain_model_path: 预训练模型路径
            freeze_encoder: 是否冻结编码器
            unfreeze_last_layers: 解冻最后几层
        """
        # 加载预训练权重
        checkpoint = torch.load(pretrain_model_path,
                                map_location='cpu', weights_only=False)

        if 'symbol_encoder' in checkpoint:
            self.transformer.symbol_encoder.load_state_dict(
                checkpoint['symbol_encoder'], strict=False
            )

        if 'transformer_encoder' in checkpoint:
            self.transformer.transformer_encoder.load_state_dict(
                checkpoint['transformer_encoder'], strict=False
            )

        # 冻结参数
        if freeze_encoder:
            self.transformer.freeze_parameters(
                freeze_embedding=True,
                freeze_encoder=True,
                unfreeze_last_layers=unfreeze_last_layers
            )

    def get_molecular_embedding(self, smiles_ids: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取分子嵌入表示

        Args:
            smiles_ids: SMILES ID序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            torch.Tensor: 分子嵌入 [batch_size, d_model]
        """
        batch_size = smiles_ids.size(0)
        device = smiles_ids.device

        # 添加CLS标记
        cls_tokens = torch.ones(
            batch_size, 1, dtype=torch.long, device=device) * SPECIAL_TOKENS.CLS_ID
        x = torch.cat([cls_tokens, smiles_ids], dim=1)

        # 创建填充掩码
        if attention_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([cls_mask, attention_mask == 0], dim=1)
        else:
            padding_mask = (x == SPECIAL_TOKENS.PAD_ID)

        # Transformer编码
        with torch.no_grad():
            encoded = self.transformer(x, padding_mask)

        # 返回CLS位置的表示
        return encoded[:, 0, :]

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        return {
            'num_tokens': self.config.num_tokens,
            'num_labels': self.config.num_labels,
            'd_model': self.config.dim_embed,
            'num_layers': self.config.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }


import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    解决类别不平衡问题
    """
    def __init__(self, alpha=1.0, gamma=2.0, pos_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weights = pos_weights
        
    def forward(self, inputs, targets):
        # 计算BCE损失
        if self.pos_weights is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=self.pos_weights, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)
        
        # 计算focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # 计算focal loss
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

class ImbalancedBCELoss(nn.Module):
    """
    结合类别权重和Focal Loss的损失函数
    """
    def __init__(self, pos_weights=None, alpha=1.0, gamma=2.0, weight_factor=1.0):
        super(ImbalancedBCELoss, self).__init__()
        self.pos_weights = pos_weights
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, pos_weights=pos_weights)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.weight_factor = weight_factor
        
    def forward(self, inputs, targets):
        # 组合BCE和Focal Loss
        bce = self.bce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        # 加权组合
        return self.weight_factor * focal + (1 - self.weight_factor) * bce


class PropertyLoss(nn.Module):
    """分子性质预测损失函数"""

    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None):
        """初始化损失函数

        Args:
            num_labels: 标签数量
            class_weights: 类别权重（用于不平衡数据）
        """
        super().__init__()

        # self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失

        Args:
            predictions: 预测结果 [batch_size, num_labels]
            targets: 目标标签 [batch_size, num_labels]

        Returns:
            torch.Tensor: 损失值
        """
        return nn.functional.binary_cross_entropy(predictions, targets.float())


class ThresholdOptimizer:
    """
    自适应阈值优化器
    为每个类别找到最优的决策阈值
    """
    def __init__(self, metric='f1', search_range=(0.1, 0.9), search_steps=50):
        self.metric = metric
        self.search_range = search_range
        self.search_steps = search_steps
        self.optimal_thresholds = None
        
    def optimize_thresholds(self, y_true, y_prob):
        """
        为每个类别优化阈值
        """
        num_classes = y_true.shape[1]
        optimal_thresholds = np.zeros(num_classes)
        
        # 为每个类别独立优化阈值
        for class_idx in range(num_classes):
            y_true_class = y_true[:, class_idx]
            y_prob_class = y_prob[:, class_idx]
            
            # 如果该类别没有正样本，使用默认阈值
            if y_true_class.sum() == 0:
                optimal_thresholds[class_idx] = 0.5
                continue
                
            best_score = -1
            best_threshold = 0.5
            
            # 搜索最优阈值
            thresholds = np.linspace(self.search_range[0], self.search_range[1], self.search_steps)
            
            for threshold in thresholds:
                y_pred_class = (y_prob_class >= threshold).astype(int)
                
                # 计算指定指标
                if self.metric == 'f1':
                    score = f1_score(y_true_class, y_pred_class, zero_division=0)
                elif self.metric == 'precision':
                    pass
                    # score = precision_score(y_true_class, y_pred_class, zero_division=0)
                elif self.metric == 'recall':
                    pass
                    # score = recall_score(y_true_class, y_pred_class, zero_division=0)
                else:
                    # 默认使用F1
                    score = f1_score(y_true_class, y_pred_class, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            optimal_thresholds[class_idx] = best_threshold
        
        self.optimal_thresholds = optimal_thresholds
        return optimal_thresholds
    
    def predict(self, y_prob):
        """
        使用优化的阈值进行预测
        """
        if self.optimal_thresholds is None:
            raise ValueError("需要先调用optimize_thresholds方法")
        predictions = np.zeros_like(y_prob)
        for class_idx in range(y_prob.shape[1]):
            predictions[:, class_idx] = (y_prob[:, class_idx] >= self.optimal_thresholds[class_idx]).astype(int)
        return predictions
    
def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """计算评估指标

    Args:
        predictions: 预测结果 [batch_size, num_labels]
        targets: 目标标签 [batch_size, num_labels]

    Returns:
        Dict[str, float]: 评估指标
    """
    metrics = {}

    # ============== 多标签分类 ==============
    print(predictions.max())    
    pred_binary = (predictions > 0.5).float()
    accuracy = (pred_binary == targets.float()).float().mean()
    metrics['accuracy'] = accuracy.item()

    # 计算F1-macro和AUC-macro
    try:
        # F1-macro（需要将标签和预测转为numpy）
        metrics['f1_macro'] = f1_score(
            targets.cpu().numpy(),
            pred_binary.cpu().numpy(),
            average='macro'
        )

        # AUC-macro（需确保预测值是概率值）
        metrics['auc_macro'] = roc_auc_score(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average='macro'
        )
    except ValueError:  # 处理某些标签全0或全1的情况
        pass


    return metrics


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'min'):
        """初始化早停机制

        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            mode: 监控模式 ('min' 或 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """检查是否应该早停

        Args:
            score: 当前分数

        Returns:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        """判断当前分数是否更好"""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:  # mode == 'max'
            return current > best + self.min_delta
