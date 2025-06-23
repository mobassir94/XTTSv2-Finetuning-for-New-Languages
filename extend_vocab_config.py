import argparse
import os
import pandas as pd
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def extend_tokenizer(args):
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")
    os.makedirs(root, exist_ok=True)

    # Read both train and eval CSVs
    train_df = pd.read_csv(args.metadata_train_path, sep="|")
    eval_df = pd.read_csv(args.metadata_eval_path, sep="|")
    texts = train_df["text"].tolist() + eval_df["text"].tolist()

    print(f"📝 Total training + eval samples: {len(texts)}")

    # Load old vocab to get size
    old_vocab_path = os.path.join(root, "vocab.json")
    with open(old_vocab_path, "r", encoding="utf-8") as f:
        old_vocab = json.load(f)
    print(f"🔹 Old vocabulary size: {len(old_vocab)}")

    # Train new tokenizer from Hindi data
    new_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    new_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=args.extended_vocab_size, special_tokens=[f"[{args.language}]", "[UNK]"])
    new_tokenizer.train_from_iterator(texts, trainer)

    # Add post-processing
    new_tokenizer.post_processor = TemplateProcessing(
        single=f"[{args.language}] $A",
        special_tokens=[(f"[{args.language}]", new_tokenizer.token_to_id(f"[{args.language}]"))]
    )

    new_vocab_size = new_tokenizer.get_vocab_size()
    print(f"🆕 New vocabulary size: {new_vocab_size}")

    # Merge old + new vocab (only useful if loading is manual later)
    final_vocab = set(list(old_vocab.keys()) + list(new_tokenizer.get_vocab().keys()))
    print(f"✅ Final merged vocabulary size: {len(final_vocab)}")

    # Save new tokenizer as tokenizer.json
    tokenizer_out_path = os.path.join(root, "tokenizer.json")
    new_tokenizer.save(tokenizer_out_path)
    print(f"📦 Saved updated tokenizer: {tokenizer_out_path}")

    # Overwrite vocab.json with the new vocab dict for compatibility
    with open(os.path.join(root, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(new_tokenizer.get_vocab(), f, ensure_ascii=False)

    # Update config.json to include the new language
    config_path = os.path.join(root, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.language not in config.get("languages", []):
        config["languages"].append(args.language)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"✅ Added '{args.language}' to config.json")
    else:
        print(f"ℹ️ Language '{args.language}' already present in config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--metadata_train_path", type=str, required=True)
    parser.add_argument("--metadata_eval_path", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--extended_vocab_size", type=int, default=2000)
    args = parser.parse_args()

    extend_tokenizer(args)
