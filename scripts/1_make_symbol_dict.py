from csv_processor import CSVProcessor
from symbol_dictionary import SymbolDictionary
import pandas as pd
import json

def make_symbol_dict():
    processor = CSVProcessor()
    sentences, labels, label_to_id, label_columns = processor.load_multi_label_csv("input/smiles.csv")
    sample = dict(valid_smiles=sentences[:5],
                  label_to_id=list(label_to_id.items())[:5],
                  label_columns=label_columns[:5])
    print(json.dumps(sample,ensure_ascii=False,indent=2))
    
    symbol_dictionary = SymbolDictionary()
    for sentence in sentences:
        symbol_dictionary.add_symbols_from_smiles(sentence)
    symbol_dictionary.finalize_dictionary()
    # print(symbol_dictionary.symbol_to_id)
    print(symbol_dictionary.id_to_symbol)
    ids = symbol_dictionary.smiles_to_ids("COCO")
    print("ids",ids)
    print("smiles", symbol_dictionary.ids_to_smiles(ids))
    print(symbol_dictionary)
    output_path = "output/symbol_dict.pkl"
    symbol_dictionary.save(output_path)
    print(f"save symbol dict to {output_path}")
    
    items = []
    for i, sentence in enumerate(sentences):
        token_ids = symbol_dictionary.smiles_to_ids(sentence)
        descriptor = []
        for label_idx, is_positive in enumerate(labels[i]):
            if is_positive:
                descriptor.append(label_columns[label_idx])
        items.append(dict(sentence=sentence, 
                          descriptor=descriptor,
                          token_ids=token_ids, 
                          label_ids=labels[i]))
        
    df = pd.DataFrame(items)
    df.to_csv("output/data.csv",index=False,header=True)
    
    with open("output/label_metadata.json",'w',encoding='utf-8') as f:
        json.dump(dict(label_to_id=label_to_id, label_columns=label_columns),f,indent=2)
    
    
if __name__ == '__main__':
    make_symbol_dict()