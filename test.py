from csv_processor import CSVProcessor
from symbol_dictionary import SymbolDictionary
import json

def a():
    processor = CSVProcessor()
    valid_smiles, valid_labels, label_to_id, label_columns = processor.load_multi_label_csv("input/smiles.csv")
    sample = dict(valid_smiles=valid_smiles[:5],
                  label_to_id=list(label_to_id.items())[:5],
                  label_columns=label_columns[:5])
    print(json.dumps(sample,ensure_ascii=False,indent=2))
    
    symbol_dictionary = SymbolDictionary()
    for smiles_str in valid_smiles:
        symbol_dictionary.add_symbols_from_smiles(smiles_str)
    symbol_dictionary.finalize_dictionary()
    # print(symbol_dictionary.symbol_to_id)
    print(symbol_dictionary.id_to_symbol)
    ids = symbol_dictionary.smiles_to_ids("COCO")
    print("ids",ids)
    print("smiles", symbol_dictionary.ids_to_smiles(ids))
    print(symbol_dictionary)
    
    
if __name__ == '__main__':
    a()