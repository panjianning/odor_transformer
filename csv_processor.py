import pandas as pd
from rdkit import Chem
from typing import Tuple, List, Dict
import os

class CSVProcessor:
    def __init__(self):
        pass
    
    def load_multi_label_csv(self, csv_path: str,
                             sentence_column: str = 'nonStereoSMILES',
                             descriptor_column: str = 'descriptors',
                             ) -> Tuple[List[str], List[List[int]], Dict[str, int], List[str]]:
        """加载多标签CSV数据

        Args:
            csv_path: CSV文件路径
            smiles_column: SMILES列名
            label_column: 描述符列名（可选）

        Returns:
            Tuple: (SMILES列表, 标签矩阵, 标签到ID映射, 标签名列表)
        """
        print(f"加载多标签CSV数据: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        print(f"原始数据形状: {df.shape}")

        # 检查必要的列
        if sentence_column not in df.columns:
            raise ValueError(f"找不到SENTENCE列: {sentence_column}")

        # 获取所有可能的标签列（除了SMILES和描述符列）
        exclude_columns = {sentence_column, descriptor_column}

        label_columns = [
            col for col in df.columns if col not in exclude_columns]
        print(f"发现 {len(label_columns)} 个标签列")

        # 创建标签到ID的映射
        label_to_id = {label: i for i, label in enumerate(label_columns)}

        # 处理数据
        sentences = []
        labels = []
        skipped_count = 0

        for idx, row in df.iterrows():
            sentence = str(row[sentence_column]).strip()

            # 验证SMILES
            if not self._is_valid_smiles(sentence):
                skipped_count += 1
                continue

            # 规范化SMILES
            try:
                mol = Chem.MolFromSmiles(sentence)
                canonical_smiles = Chem.MolToSmiles(mol)
            except:
                skipped_count += 1
                continue

            # 构建标签向量
            label_vector = [0] * len(label_columns)
            for i, label in enumerate(label_columns):
                if label in df.columns:
                    value = row[label]
                    # 处理不同的标签格式
                    if pd.isna(value):
                        label_vector[i] = 0
                    elif isinstance(value, (int, float)):
                        label_vector[i] = int(value > 0.5)  # 阈值化
                    else:
                        label_vector[i] = 1 if str(value).lower() in [
                            '1', 'true', 'yes'] else 0

            sentences.append(canonical_smiles)
            labels.append(label_vector)

        print(f"处理完成:")
        print(f"  有效样本: {len(sentences)}")
        print(f"  跳过样本: {skipped_count}")
        print(f"  标签维度: {len(label_columns)}")

        return sentences, labels, label_to_id, label_columns

    def _is_valid_smiles(self, smiles: str) -> bool:
        """验证SMILES有效性

        Args:
            smiles: SMILES字符串

        Returns:
            bool: 是否有效
        """
        if not smiles:
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False