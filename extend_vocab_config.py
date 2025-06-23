import argparse
import os
import json
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def combine_tokenizers(old_tokenizer_path, new_tokenizer_path, save_dir):
    with open(os.path.join(old_tokenizer_path, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab1 = json.load(f)
    with open(os.path.join(new_tokenizer_path, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab2 = json.load(f)

    merged_vocab = {}
    idx = 0
    for word in vocab1:
        if word not in merged_vocab:
            merged_vocab[word] = idx
            idx += 1
    for word in vocab2:
        if word not in merged_vocab:
            merged_vocab[word] = idx
            idx += 1

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_vocab, f, ensure_ascii=False, indent=2)

    merges_1 = os.path.join(old_tokenizer_path, 'merges.txt')
    merges_2 = os.path.join(new_tokenizer_path, 'merges.txt')
    merged_merges = os.path.join(save_dir, 'merges.txt')

    with open(merged_merges, 'w', encoding='utf-8') as out_file:
        with open(merges_1, 'r', encoding='utf-8') as f1:
            out_file.writelines(f1.readlines())
        with open(merges_2, 'r', encoding='utf-8') as f2:
            lines = f2.readlines()[1:]
            out_file.writelines(lines)


def extend_tokenizer(args):
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files")
    vocab_json_path = os.path.join(root, "vocab.json")

    if not os.path.exists(vocab_json_path):
        raise FileNotFoundError(f"Original vocab.json not found at: {vocab_json_path}")

    old_tokenizer_path = os.path.join(root, "old_tokenizer")
    os.makedirs(old_tokenizer_path, exist_ok=True)

    tokenizer = Tokenizer.from_file(vocab_json_path)
    tokenizer.model.save(old_tokenizer_path)

    with open(os.path.join(old_tokenizer_path, "vocab.json"), "r", encoding="utf-8") as f:
        old_vocab = json.load(f)
    print(f"🔹 Old vocabulary size: {len(old_vocab)}")

    # ✅ Load both train and eval CSVs
    df_train = pd.read_csv(args.metadata_path, sep="|")
    df_eval = pd.read_csv(args.metadata_eval_path, sep="|")
    texts = df_train["text"].astype(str).tolist() + df_eval["text"].astype(str).tolist()

    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=[f"[{args.language}]"], vocab_size=args.extended_vocab_size)
    new_tokenizer.train_from_iterator(texts, trainer=trainer)

    new_tokenizer_path = os.path.join(root, "new_tokenizer")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)

    with open(os.path.join(new_tokenizer_path, "vocab.json"), "r", encoding="utf-8") as f:
        new_vocab = json.load(f)
    print(f"🆕 New vocabulary size: {len(new_vocab)}")

    merged_tokenizer_path = os.path.join(root, "merged_tokenizer")
    combine_tokenizers(old_tokenizer_path, new_tokenizer_path, merged_tokenizer_path)

    with open(os.path.join(merged_tokenizer_path, "vocab.json"), "r", encoding='utf-8') as f:
        final_vocab = json.load(f)
    print(f"✅ Final merged vocabulary size: {len(final_vocab)}")

    os.replace(os.path.join(merged_tokenizer_path, "vocab.json"), os.path.join(root, "vocab.json"))
    os.replace(os.path.join(merged_tokenizer_path, "merges.txt"), os.path.join(root, "merges.txt"))

    for path in [old_tokenizer_path, new_tokenizer_path, merged_tokenizer_path]:
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
            os.rmdir(path)


def adjust_config(args):
    config_path = os.path.join(args.output_path, "XTTS_v2.0_original_model_files", "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.language not in config.get("languages", []):
        config["languages"].append(args.language)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Language '{args.language}' added to config.json")
    else:
        print(f"ℹ️ Language '{args.language}' already present in config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Path to XTTS model dir")
    parser.add_argument("--metadata_path", type=str, required=True, help="Train metadata CSV")
    parser.add_argument("--metadata_eval_path", type=str, required=True, help="Eval metadata CSV")
    parser.add_argument("--language", type=str, required=True, help="Language ID, e.g., hi")
    parser.add_argument("--extended_vocab_size", type=int, default=2000, help="Size of new vocab")

    args = parser.parse_args()
    extend_tokenizer(args)
    adjust_config(args)
