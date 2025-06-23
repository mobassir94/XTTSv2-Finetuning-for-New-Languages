import argparse
from tokenizers import Tokenizer
import os
import pandas as pd
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import json

def combine_tokenizers(old_tokenizer, new_tokenizer, save_dir):
    # Load both the json files, take the union, and store it
    json1 = json.load(open(os.path.join(old_tokenizer, 'vocab.json')))
    json2 = json.load(open(os.path.join(new_tokenizer, 'vocab.json')))
    
    # Create a new vocabulary
    new_vocab = {}
    idx = 0
    for word in json1.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1
    
    # Add words from second tokenizer
    for word in json2.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1
    
    # Make the directory if necessary
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the vocab
    with open(os.path.join(save_dir, 'vocab.json'), 'w') as fp:
        json.dump(new_vocab, fp, ensure_ascii=False)
    
    # Merge the two merges file. Don't handle duplicates here
    # Concatenate them, but ignore the first line of the second file
    os.system('cat {} > {}'.format(os.path.join(old_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))
    os.system('tail -n +2 -q {} >> {}'.format(os.path.join(new_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))

def load_texts_from_csvs(csv_paths):
    """Load and combine text data from multiple CSV files"""
    all_texts = []
    
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            print(f"Loading texts from: {csv_path}")
            df = pd.read_csv(csv_path, sep="|")
            texts = df.text.to_list()
            all_texts.extend(texts)
            print(f"  - Loaded {len(texts)} texts")
        else:
            print(f"Warning: CSV file not found: {csv_path}")
    
    print(f"Total texts loaded: {len(all_texts)}")
    return all_texts

def extend_tokenizer(args):
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")
    
    # Load existing tokenizer and get vocab size
    existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    old_vocab_size = existing_tokenizer.get_vocab_size()
    print(f"Old vocabulary size: {old_vocab_size}")
    
    # save seperately vocab, merges
    old_tokenizer_path = os.path.join(root, "old_tokenizer/")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    existing_tokenizer.model.save(old_tokenizer_path)
    
    # Collect CSV paths
    csv_paths = []
    if args.train_metadata_path:
        csv_paths.append(args.train_metadata_path)
    if args.eval_metadata_path:
        csv_paths.append(args.eval_metadata_path)
    
    # Fallback to single metadata_path if provided (backward compatibility)
    if hasattr(args, 'metadata_path') and args.metadata_path:
        csv_paths.append(args.metadata_path)
    
    if not csv_paths:
        raise ValueError("No CSV paths provided. Use --train_metadata_path and/or --eval_metadata_path")
    
    # Load texts from all CSV files
    texts = load_texts_from_csvs(csv_paths)
    
    if not texts:
        raise ValueError("No texts loaded from CSV files")
    
    # train new tokenizer
    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=[f"[{args.language}]"], vocab_size=args.extended_vocab_size)
    new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
    new_tokenizer.add_special_tokens([f"[{args.language}]"])
    
    new_tokenizer_path = os.path.join(root, "new_tokenizer/")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)
    
    merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")
    combine_tokenizers(
        old_tokenizer_path,
        new_tokenizer_path,
        merged_tokenizer_path
    )
    
    tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    tokenizer.model = tokenizer.model.from_file(os.path.join(merged_tokenizer_path, 'vocab.json'), os.path.join(merged_tokenizer_path, 'merges.txt'))
    tokenizer.add_special_tokens([f"[{args.language}]"])
    tokenizer.save(os.path.join(root, "vocab.json"))
    
    # Get new vocab size and print comparison
    new_vocab_size = tokenizer.get_vocab_size()
    print(f"New vocabulary size: {new_vocab_size}")
    print(f"Vocabulary increased by: {new_vocab_size - old_vocab_size} tokens")
    
    os.system(f'rm -rf {old_tokenizer_path} {new_tokenizer_path} {merged_tokenizer_path}')

def adjust_config(args):
    config_path = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["languages"] += [args.language]
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Output path for model files")
    parser.add_argument("--train_metadata_path", type=str, help="Path to train.csv file")
    parser.add_argument("--eval_metadata_path", type=str, help="Path to eval.csv file")
    parser.add_argument("--metadata_path", type=str, help="Single CSV path (backward compatibility)")
    parser.add_argument("--language", type=str, required=True, help="Language code")
    parser.add_argument("--extended_vocab_size", default=2000, type=int, required=True, help="Extended vocabulary size")
    
    args = parser.parse_args()
    extend_tokenizer(args)
    adjust_config(args)
