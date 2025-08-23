"""优化的Transformer模型

该模块实现使用DeepChem特征增强的Transformer模型。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
import deepchem as dc
from deepchem.feat import CircularFingerprint, Mol2VecFeaturizer

from .transformer_model import BaseTransformerModel
from symbol_dictionary import SPECIAL_TOKENS

class EnhancedTransformerModel(nn.Module):
    """增强的Transformer模型
    
    结合DeepChem分子特征的优化Transformer模型
    """
    
    def __init__(self, num_tokens: int, d_model: int, num_heads: int,
                 d_hidden: int, num_layers: int, num_labels: int,
                 feature_dim: int = 1024, fusion_dim: int = 512,
                 dropout: float = 0.1, use_fingerprint: bool = True,
                 use_mol2vec: bool = False):
        """初始化增强的Transformer模型
        
        Args:
            num_tokens: 词汇表大小
            d_model: 模型维度
            num_heads: 多头注意力头数
            d_hidden: 前馈网络隐藏层维度
            num_layers: 编码器层数
            num_labels: 标签数量
            feature_dim: 特征维度
            fusion_dim: 融合层维度
            dropout: Dropout率
            use_fingerprint: 是否使用分子指纹特征
            use_mol2vec: 是否使用Mol2Vec特征
        """
        super().__init__()
        
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.fusion_dim = fusion_dim
        self.use_fingerprint = use_fingerprint
        self.use_mol2vec = use_mol2vec
        
        # 基础Transformer模型
        self.transformer = BaseTransformerModel(
            num_tokens=num_tokens,
            d_model=d_model,
            num_heads=num_heads,
            d_hidden=d_hidden,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 分子特征提取器
        if use_fingerprint:
            self.fingerprint_featurizer = CircularFingerprint(size=feature_dim)
        
        if use_mol2vec:
            try:
                self.mol2vec_featurizer = Mol2VecFeaturizer()
            except:
                self.use_mol2vec = False
                print("Mol2Vec特征提取器不可用，已禁用")
        
        # 特征投影层
        total_feature_dim = d_model
        if use_fingerprint:
            total_feature_dim += feature_dim
        if use_mol2vec:
            total_feature_dim += 300  # Mol2Vec特征维度
        
        self.feature_projection = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_labels)
        )
        
        self.output_activation = nn.Sigmoid()
    
    def extract_molecular_features(self, smiles_list: List[str]) -> torch.Tensor:
        """提取分子特征
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            torch.Tensor: 分子特征张量
        """
        features = []
        
        for smiles in smiles_list:
            mol_features = []
            
            # 提取分子指纹特征
            if self.use_fingerprint:
                try:
                    fp = self.fingerprint_featurizer.featurize([smiles])
                    if fp[0] is not None:
                        mol_features.extend(fp[0])
                    else:
                        mol_features.extend(np.zeros(self.feature_dim))
                except:
                    mol_features.extend(np.zeros(self.feature_dim))
            
            # 提取Mol2Vec特征
            if self.use_mol2vec:
                try:
                    mv = self.mol2vec_featurizer.featurize([smiles])
                    if mv[0] is not None:
                        mol_features.extend(mv[0])
                    else:
                        mol_features.extend(np.zeros(300))
                except:
                    mol_features.extend(np.zeros(300))
            
            features.append(mol_features)
        
        return torch.tensor(np.array(features), dtype=torch.float32)
    
    def forward(self, src: torch.Tensor, smiles_list: List[str],
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            src: 输入符号ID序列 [batch_size, seq_len]
            smiles_list: 对应的SMILES字符串列表
            padding_mask: 填充掩码 [batch_size, seq_len]
            
        Returns:
            torch.Tensor: 预测结果 [batch_size, num_labels]
        """
        batch_size = src.size(0)
        device = src.device
        
        # 添加CLS标记
        cls_tokens = torch.ones(
            batch_size, 1, dtype=torch.long, device=device) * SPECIAL_TOKENS.CLS_ID
        x = torch.cat([cls_tokens, src], dim=1)
        
        # 创建填充掩码
        if padding_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([cls_mask, padding_mask == 0], dim=1)
        else:
            padding_mask = (x == SPECIAL_TOKENS.PAD_ID)
        
        # Transformer编码
        encoded = self.transformer(x, padding_mask)
        cls_representation = encoded[:, 0, :]  # [batch_size, d_model]
        
        # 提取分子特征
        molecular_features = self.extract_molecular_features(smiles_list).to(device)
        
        # 特征融合
        combined_features = torch.cat([cls_representation, molecular_features], dim=1)
        fused_features = self.feature_projection(combined_features)
        
        # 分类
        logits = self.classifier(fused_features)
        output = self.output_activation(logits)
        
        return output

class AttentionEnhancedTransformer(nn.Module):
    """注意力机制增强的Transformer模型
    
    使用改进的注意力机制和特征融合策略
    """
    
    def __init__(self, num_tokens: int, d_model: int, num_heads: int,
                 d_hidden: int, num_layers: int, num_labels: int,
                 dropout: float = 0.1):
        """初始化注意力增强的Transformer模型"""
        super().__init__()
        
        self.transformer = BaseTransformerModel(
            num_tokens=num_tokens,
            d_model=d_model,
            num_heads=num_heads,
            d_hidden=d_hidden,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 多头注意力池化层
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels)
        )
        
        self.output_activation = nn.Sigmoid()
    
    def forward(self, src: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        batch_size = src.size(0)
        device = src.device
        
        # 添加CLS标记
        cls_tokens = torch.ones(
            batch_size, 1, dtype=torch.long, device=device) * SPECIAL_TOKENS.CLS_ID
        x = torch.cat([cls_tokens, src], dim=1)
        
        # 创建填充掩码
        if padding_mask is not None:
            cls_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([cls_mask, padding_mask == 0], dim=1)
        else:
            padding_mask = (x == SPECIAL_TOKENS.PAD_ID)
        
        # Transformer编码
        encoded = self.transformer(x, padding_mask)
        
        # CLS表示
        cls_representation = encoded[:, 0, :]  # [batch_size, d_model]
        
        # 注意力池化
        attn_output, attn_weights = self.attention_pooling(
            cls_representation.unsqueeze(1),  # query
            encoded,  # key
            encoded,  # value
            key_padding_mask=padding_mask
        )
        
        attn_output = attn_output.squeeze(1)  # [batch_size, d_model]
        
        # 特征融合
        combined_features = torch.cat([cls_representation, attn_output], dim=1)
        
        # 分类
        logits = self.classifier(combined_features)
        output = self.output_activation(logits)
        
        return output

def create_optimized_transformer_config() -> dict:
    """创建优化的Transformer配置"""
    return {
        'num_tokens': 306,
        'd_model': 256,  # 增加模型维度
        'num_heads': 8,
        'd_hidden': 1024,  # 增加隐藏层维度
        'num_layers': 8,  # 增加层数
        'num_labels': 138,
        'feature_dim': 2048,  # 更大的特征维度
        'fusion_dim': 512,
        'dropout': 0.2,  # 更高的dropout防止过拟合
        'use_fingerprint': True,
        'use_mol2vec': False
    }

# 工具函数
def get_molecule_descriptors(smiles: str) -> dict:
    """获取分子描述符"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return {
                'mol_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_h_donors': Descriptors.NumHDonors(mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            }
    except:
        pass
    
    return {}
