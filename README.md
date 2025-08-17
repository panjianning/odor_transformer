``` bash
python -m scripts.1_make_symbol_dict
```


FocalLoss
```

Epoch 50/50
--------------------------------------------------
  Batch 0/125, Loss: 0.5592
  Batch 50/125, Loss: 0.6011
  Batch 100/125, Loss: 0.6523
tensor(1.0000)
tensor(1.0000)
训练结果:
  Train Loss: 0.6304
  Val Loss: 0.6384
  Train Acc: 0.8983
  Val Acc: 0.8914
  Train F1-Macro: 0.2204
  Val F1-Macro: 0.1971
  Train AUC-Macro: 0.8298
  Val AUC-Macro: 0.8187
  Time: 63.45s

训练完成! 总用时: 3155.14s

在测试集上评估...
tensor(1.0000)
测试结果:
  accuracy: 0.8916
  f1_macro: 0.1983
  auc_macro: 0.8260
  loss: 0.6424
{'accuracy': 0.8916093111038208, 'f1_macro': 0.19832468874378537, 'auc_macro': 0.8260055052983379, 'loss': 0.6423734687268734}
```


## 不用hf tokenizer
```

在测试集上评估...
tensor(0.9999)
测试结果:
  accuracy: 0.8941
  f1_macro: 0.1442
  auc_macro: 0.7520
  loss: 0.6644
{'accuracy': 0.8941215872764587, 'f1_macro': 0.1442432739182713, 'auc_macro': 0.7519886806931402, 'loss': 0.6643555089831352}
```


## hf tokenizer, focal loss 1.0

```
Epoch 100/100
--------------------------------------------------
  Batch 0/125, Loss: 0.4093
  Batch 50/125, Loss: 0.4885
  Batch 100/125, Loss: 0.5121
tensor(1.0000)
tensor(1.0000)
训练结果:
  Train Loss: 0.4768
  Val Loss: 0.4833
  Train Acc: 0.8859
  Val Acc: 0.8783
  Train F1-Macro: 0.2069
  Val F1-Macro: 0.1947
  Train AUC-Macro: 0.8300
  Val AUC-Macro: 0.8247
  Time: 36.57s

训练完成! 总用时: 3872.49s

在测试集上评估...
tensor(1.0000)
测试结果:
  accuracy: 0.8763
  f1_macro: 0.1914
  auc_macro: 0.8282
  loss: 0.4851
{'accuracy': 0.8763033151626587, 'f1_macro': 0.19139661480346407, 'auc_macro': 0.8281650261676223, 'loss': 0.4850555583834648}
```