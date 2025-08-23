"""图神经网络模型实现

该模块实现基于DeepChem的图神经网络模型，用于与Transformer模型对比。
"""

import torch
import torch.nn as nn
import deepchem as dc
from deepchem.models.torch_models import GraphConvModel, MPNNModel, AttentiveFPModel
from deepchem.feat import MolGraphConvFeaturizer, CircularFingerprint
from typing import Dict, List, Optional
import numpy as np

class GraphModelWrapper:
    """图神经网络模型包装器
    
    封装DeepChem的图模型，提供统一的接口用于训练和评估。
    """
    
    def __init__(self, model_type: str = 'GraphConv', 
                 n_tasks: int = 138,
                 mode: str = 'classification',
                 **kwargs):
        """初始化图模型
        
        Args:
            model_type: 模型类型 ('GraphConv', 'MPNN', 'AttentiveFP')
            n_tasks: 任务数量
            mode: 模式 ('classification' 或 'regression')
            **kwargs: 模型参数
        """
        self.model_type = model_type
        self.n_tasks = n_tasks
        self.mode = mode
        self.model = None
        self.featurizer = None
        
        # 设置特征提取器
        if model_type in ['GraphConv', 'MPNN']:
            self.featurizer = MolGraphConvFeaturizer()
        elif model_type == 'AttentiveFP':
            self.featurizer = MolGraphConvFeaturizer(use_edges=True)
        
        # 初始化模型
        self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        """初始化具体的图模型"""
        model_params = {
            'n_tasks': self.n_tasks,
            'mode': self.mode,
            **kwargs
        }
        
        if self.model_type == 'GraphConv':
            self.model = GraphConvModel(**model_params)
        elif self.model_type == 'MPNN':
            self.model = MPNNModel(**model_params)
        elif self.model_type == 'AttentiveFP':
            self.model = AttentiveFPModel(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def featurize_data(self, smiles_list: List[str]) -> np.ndarray:
        """将SMILES列表转换为图特征
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            np.ndarray: 图特征数组
        """
        if not self.featurizer:
            raise ValueError("未设置特征提取器")
        
        features = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                feat = self.featurizer.featurize([smiles])
                if feat[0] is not None:
                    features.append(feat[0])
                    valid_indices.append(i)
            except:
                continue
        
        return np.array(features), valid_indices
    
    def train(self, train_smiles: List[str], train_labels: List[List[float]],
              val_smiles: List[str] = None, val_labels: List[List[float]] = None,
              epochs: int = 100, **kwargs):
        """训练模型
        
        Args:
            train_smiles: 训练集SMILES
            train_labels: 训练集标签
            val_smiles: 验证集SMILES
            val_labels: 验证集标签
            epochs: 训练轮数
        """
        # 特征化训练数据
        train_features, train_idx = self.featurize_data(train_smiles)
        train_labels = np.array(train_labels)[train_idx]
        
        # 创建DeepChem数据集
        dataset = dc.data.NumpyDataset(train_features, train_labels)
        
        # 训练模型
        self.model.fit(dataset, nb_epoch=epochs, **kwargs)
        
        # 如果有验证集，进行验证
        if val_smiles and val_labels:
            val_features, val_idx = self.featurize_data(val_smiles)
            val_labels = np.array(val_labels)[val_idx]
            val_dataset = dc.data.NumpyDataset(val_features, val_labels)
            
            # 评估验证集
            metrics = self.evaluate(val_dataset)
            return metrics
        
        return None
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """预测
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            np.ndarray: 预测结果
        """
        features, valid_indices = self.featurize_data(smiles_list)
        dataset = dc.data.NumpyDataset(features)
        
        predictions = self.model.predict(dataset)
        return predictions, valid_indices
    
    def evaluate(self, dataset: dc.data.Dataset) -> Dict[str, float]:
        """评估模型
        
        Args:
            dataset: DeepChem数据集
            
        Returns:
            Dict[str, float]: 评估指标
        """
        metrics = {}
        
        if self.mode == 'classification':
            # 分类任务评估指标
            from deepchem.metrics import roc_auc_score, accuracy_score
            
            predictions = self.model.predict(dataset)
            true_labels = dataset.y
            
            try:
                metrics['auc'] = roc_auc_score(true_labels, predictions)
                metrics['accuracy'] = accuracy_score(true_labels, predictions > 0.5)
            except:
                metrics['auc'] = 0.0
                metrics['accuracy'] = 0.0
        
        return metrics

class MultiModalTransformer(nn.Module):
    """多模态Transformer模型
    
    结合Transformer序列表示和图特征的多模态模型
    """
    
    def __init__(self, transformer_model, graph_feature_dim: int = 1024,
                 fusion_dim: int = 512, num_labels: int = 138):
        """初始化多模态模型
        
        Args:
            transformer_model: 预训练的Transformer模型
            graph_feature_dim: 图特征维度
            fusion_dim: 融合层维度
            num_labels: 标签数量
        """
        super().__init__()
        
        self.transformer = transformer_model
        self.graph_feature_dim = graph_feature_dim
        
        # 图特征处理层
        self.graph_projection = nn.Linear(graph_feature_dim, fusion_dim)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(transformer_model.d_model + fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU()
        )
        
        # 分类头
        self.classifier = nn.Linear(fusion_dim // 2, num_labels)
        self.output_activation = nn.Sigmoid()
        
        # 图特征提取器
        self.fingerprint_featurizer = CircularFingerprint(size=graph_feature_dim)
    
    def extract_graph_features(self, smiles_list: List[str]) -> torch.Tensor:
        """提取分子图特征
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            torch.Tensor: 图特征张量
        """
        features = []
        for smiles in smiles_list:
            try:
                feat = self.fingerprint_featurizer.featurize([smiles])
                features.append(feat[0])
            except:
                # 如果特征提取失败，使用零向量
                features.append(np.zeros(self.graph_feature_dim))
        
        return torch.tensor(np.array(features), dtype=torch.float32)
    
    def forward(self, smiles_ids: torch.Tensor, 
                smiles_list: List[str],
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            smiles_ids: SMILES ID序列
            smiles_list: 对应的SMILES字符串列表
            attention_mask: 注意力掩码
            
        Returns:
            torch.Tensor: 预测结果
        """
        batch_size = smiles_ids.size(0)
        device = smiles_ids.device
        
        # Transformer编码
        transformer_output = self.transformer(smiles_ids, attention_mask)
        cls_representation = transformer_output[:, 0, :]  # [batch_size, d_model]
        
        # 提取图特征
        graph_features = self.extract_graph_features(smiles_list).to(device)
        graph_features = self.graph_projection(graph_features)  # [batch_size, fusion_dim]
        
        # 特征融合
        combined_features = torch.cat([cls_representation, graph_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # 分类
        logits = self.classifier(fused_features)
        output = self.output_activation(logits)
        
        return output
