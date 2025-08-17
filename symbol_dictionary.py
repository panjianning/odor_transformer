from typing import Dict
import pickle
import os
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModel


@dataclass
class SpecialTokens:
    """特殊标记配置"""
    PAD_ID: int = 0      # 填充标记
    CLS_ID: int = 1      # 分类标记
    BOS_ID: int = 2      # 开始标记
    EOS_ID: int = 3      # 结束标记
    MSK_ID: int = 4      # 掩码标记

    PAD_TOKEN: str = ' '
    CLS_TOKEN: str = '[CLS]'
    BOS_TOKEN: str = '[BOS]'
    EOS_TOKEN: str = '[EOS]'
    MSK_TOKEN: str = '▪'
    
SPECIAL_TOKENS = SpecialTokens()

"""符号字典管理器

该模块负责管理SMILES字符串中的符号到ID的映射关系。
主要功能：
1. 构建符号字典
2. 符号与ID的相互转换
3. 字典的保存和加载
"""

import pickle
import os
from typing import Dict, List, Optional, Tuple
from rdkit import Chem


class SymbolDictionary:
    """符号字典管理器

    管理SMILES字符串中符号与ID之间的映射关系。
    包含特殊标记（PAD, CLS, BOS, EOS, MSK）和化学符号。
    """

    def __init__(self):
        """初始化符号字典"""
        self.symbol_to_id: Dict[str, int] = {}
        self.id_to_symbol: List[str] = []
        self.next_id: int = 0
        self._initialize_special_tokens()
        
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

        SPECIAL_TOKENS.PAD_ID = self.tokenizer.pad_token_id
        SPECIAL_TOKENS.CLS_ID = self.tokenizer.cls_token_id

    def _initialize_special_tokens(self):
        """初始化特殊标记"""
        special_tokens = [
            (SPECIAL_TOKENS.PAD_TOKEN, SPECIAL_TOKENS.PAD_ID),
            (SPECIAL_TOKENS.CLS_TOKEN, SPECIAL_TOKENS.CLS_ID),
            (SPECIAL_TOKENS.BOS_TOKEN, SPECIAL_TOKENS.BOS_ID),
            (SPECIAL_TOKENS.EOS_TOKEN, SPECIAL_TOKENS.EOS_ID),
            (SPECIAL_TOKENS.MSK_TOKEN, SPECIAL_TOKENS.MSK_ID),
            ('H', 5)  # 氢原子特殊处理
        ]

        for token, token_id in special_tokens:
            self.symbol_to_id[token] = token_id
            # 确保id_to_symbol列表足够长
            while len(self.id_to_symbol) <= token_id:
                self.id_to_symbol.append('')
            self.id_to_symbol[token_id] = token

        self.next_id = 6

    def add_symbols_from_smiles(self, smiles_str: str) -> bool:
        """从SMILES字符串中提取并添加符号

        Args:
            smiles_str: SMILES字符串

        Returns:
            bool: 是否成功添加（如果SMILES无效则返回False）
        """
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return False

            # 添加原子符号
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if atom.GetIsAromatic():
                    symbol = symbol.lower()

                if symbol not in self.symbol_to_id:
                    self.symbol_to_id[symbol] = self.next_id
                    self.id_to_symbol.append(symbol)
                    self.next_id += 1
                    
            # symbols = self.tokenizer.tokenize(smiles_str)
            # for symbol in symbols:
            #     if symbol not in self.symbol_to_id:
            #         self.symbol_to_id[symbol] = self.next_id
            #         self.id_to_symbol.append(symbol)
            #         self.next_id += 1                

            return True

        except Exception:
            return False

    def finalize_dictionary(self):
        """完成字典构建，添加常用化学符号"""
        # 添加常用符号
        common_symbols = '()[]+-.\\/=#@'
        for symbol in common_symbols:
            if symbol not in self.symbol_to_id:
                self.symbol_to_id[symbol] = self.next_id
                self.id_to_symbol.append(symbol)
                self.next_id += 1

        # 添加@@符号
        if '@@' not in self.symbol_to_id:
            self.symbol_to_id['@@'] = self.next_id
            self.id_to_symbol.append('@@')
            self.next_id += 1

        # 添加数字1-9
        for i in range(1, 10):
            symbol = str(i)
            if symbol not in self.symbol_to_id:
                self.symbol_to_id[symbol] = self.next_id
                self.id_to_symbol.append(symbol)
                self.next_id += 1

        # 添加%10-%49（环标记）
        for i in range(10, 50):
            symbol = f'%{i}'
            if symbol not in self.symbol_to_id:
                self.symbol_to_id[symbol] = self.next_id
                self.id_to_symbol.append(symbol)
                self.next_id += 1

    def smiles_to_ids(self, smiles_str: str) -> List[int]:
        """将SMILES字符串转换为ID列表

        Args:
            smiles_str: SMILES字符串

        Returns:
            List[int]: ID列表
        """
        return self.tokenizer.encode(
            smiles_str,
            add_special_tokens=True,  # 添加[CLS]和[SEP]
            max_length=512,
            truncation=True,
            padding=False
        )
        ids = []
        i = 0
        max_symbol_length = max(
            len(s) for s in self.id_to_symbol) if self.id_to_symbol else 1

        while i < len(smiles_str):
            found = False
            # 从最长符号开始匹配
            for length in range(max_symbol_length, 0, -1):
                if i + length <= len(smiles_str):
                    symbol = smiles_str[i:i+length]
                    if symbol in self.symbol_to_id:
                        ids.append(self.symbol_to_id[symbol])
                        i += length
                        found = True
                        break

            if not found:
                # 未知符号用掩码标记替代
                ids.append(SPECIAL_TOKENS.MSK_ID)
                i += 1

        return ids

    def ids_to_smiles(self, ids: List[int]) -> str:
        """将ID列表转换为SMILES字符串

        Args:
            ids: ID列表

        Returns:
            str: SMILES字符串
        """
        symbols = []
        for id_val in ids:
            if 0 <= id_val < len(self.id_to_symbol):
                symbols.append(self.id_to_symbol[id_val])
            else:
                symbols.append(SPECIAL_TOKENS.MSK_TOKEN)

        return ''.join(symbols)

    def save(self, filepath: str):
        """保存符号字典到文件

        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = [self.symbol_to_id, self.id_to_symbol, self.next_id]
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """从文件加载符号字典

        Args:
            filepath: 文件路径
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.symbol_to_id, self.id_to_symbol, self.next_id = data

    @property
    def vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.next_id

    @property
    def max_symbol_length(self) -> int:
        """获取最长符号的长度"""
        return max(len(s) for s in self.id_to_symbol) if self.id_to_symbol else 1

    def get_symbol_info(self) -> Dict[str, any]:
        """获取符号字典信息"""
        return {
            'vocab_size': self.vocab_size,
            'max_symbol_length': self.max_symbol_length,
            'num_symbols': len(self.symbol_to_id),
            'special_tokens': {
                'PAD': SPECIAL_TOKENS.PAD_ID,
                'CLS': SPECIAL_TOKENS.CLS_ID,
                'BOS': SPECIAL_TOKENS.BOS_ID,
                'EOS': SPECIAL_TOKENS.EOS_ID,
                'MSK': SPECIAL_TOKENS.MSK_ID,
            }
        }

    def __len__(self) -> int:
        """返回词汇表大小"""
        return self.vocab_size

    def __str__(self) -> str:
        """返回字典信息的字符串表示"""
        info = self.get_symbol_info()
        return f"SymbolDictionary(vocab_size={info['vocab_size']}, max_length={info['max_symbol_length']})"
