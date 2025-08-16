import torch
from transformers import AutoTokenizer, AutoModel

model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

text = "ZnO"  # 注意：ChemBERTa 是在 SMILES 上训练的，最好提供 SMILES（见下方备注）
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)  # outputs.last_hidden_state: (batch, seq_len, hidden_size)

# 逐 token 的上下文化向量
token_embeddings = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)

# 句子级别向量：对 padding 位置做遮罩的平均池化
attn = inputs["attention_mask"].unsqueeze(-1)  # shape: (1, seq_len, 1)
sum_emb = (token_embeddings * attn).sum(dim=1)
lengths = attn.sum(dim=1)  # shape: (1, 1)
sentence_embedding = sum_emb / lengths  # shape: (1, hidden_size)

# 可选：L2 归一化
sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

print(sentence_embedding.shape)