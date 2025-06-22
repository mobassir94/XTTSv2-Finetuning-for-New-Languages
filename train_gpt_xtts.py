import os
import gc

import torch
import torchaudio
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# Clear GPU cache
torch.cuda.empty_cache()

# Ensure deterministic behavior
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hack - Model manager
model_manager = ModelManager()

# Define the path where XTTS v2.0 files will be downloaded
XTTS_V2_DIR = "checkpoints/XTTS_v2.0_original_model_files/"
os.makedirs(XTTS_V2_DIR, exist_ok=True)

# Download XTTS v2 if not already present
if not os.path.exists(os.path.join(XTTS_V2_DIR, "config.json")):
    print("Downloading XTTS v2.0...")
    model_path, config_path, model_item = model_manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    
    # Move files to XTTS_V2_DIR
    import shutil
    for file in ["config.json", "model.pth", "vocab.json"]:
        src = os.path.join(os.path.dirname(config_path), file)
        dst = os.path.join(XTTS_V2_DIR, file)
        if os.path.exists(src):
            shutil.copy(src, dst)
    
    # Also copy other necessary files if present
    for file in ["dvae.pth", "mel_stats.pth"]:
        src = os.path.join(os.path.dirname(config_path), file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(XTTS_V2_DIR, file))

# Define paths to model files
TOKENIZER_FILE = os.path.join(XTTS_V2_DIR, "vocab.json")
XTTS_CHECKPOINT = os.path.join(XTTS_V2_DIR, "model.pth")
DVAE_CHECKPOINT = os.path.join(XTTS_V2_DIR, "dvae.pth")
MEL_NORM_FILE = os.path.join(XTTS_V2_DIR, "mel_stats.pth")

# Ensure all required files exist
assert os.path.exists(TOKENIZER_FILE), f"Tokenizer file not found: {TOKENIZER_FILE}"
assert os.path.exists(XTTS_CHECKPOINT), f"XTTS checkpoint not found: {XTTS_CHECKPOINT}"

# Set constants
SAMPLE_RATE = 22050
DVAE_SAMPLE_RATE = 22050

# Training Parameters
BATCH_SIZE = 2
GRAD_ACUMM_STEPS = 84
EPOCHS = 50
NUM_WORKERS = 8
LR = 5e-06
REG_LOSS_ALPHA = 100
SAVE_STEP = 5000

# Define here the output folder for the dataset
OUT_PATH = "checkpoints/"

# Define your datasets to train here
DATASETS_CONFIG_LIST = []

def set_config_from_cli_args(args):
    global BATCH_SIZE, GRAD_ACUMM_STEPS, OUT_PATH, DATASETS_CONFIG_LIST
    global EPOCHS, NUM_WORKERS, LR, REG_LOSS_ALPHA, SAVE_STEP
    
    if args.output_path:
        OUT_PATH = args.output_path
    
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    
    if args.grad_acumm:
        GRAD_ACUMM_STEPS = args.grad_acumm
    
    if args.num_epochs:
        EPOCHS = args.num_epochs
    
    if args.num_workers:
        NUM_WORKERS = args.num_workers
    
    if args.lr:
        LR = args.lr
    
    if args.save_step:
        SAVE_STEP = args.save_step
    
    if args.weight_decay:
        REG_LOSS_ALPHA = args.weight_decay
    
    if args.metadatas:
        metadata_configs = args.metadatas.split(",")
        
        if len(metadata_configs) == 3:
            train_csv, eval_csv, language = metadata_configs
            dataset_config = BaseDatasetConfig(
                formatter="ljspeech",
                dataset_name=language,
                path="./",
                meta_file_train=train_csv,
                meta_file_val=eval_csv,
                language=language
            )
            DATASETS_CONFIG_LIST.append(dataset_config)
        else:
            raise ValueError("metadatas should be in format: train_csv,eval_csv,language")
    
    # Set max text and audio length from args
    global max_text_length, max_audio_length
    max_text_length = args.max_text_length
    max_audio_length = args.max_audio_length

# Configure arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--grad_acumm", type=int, default=None)
parser.add_argument("--metadatas", type=str, default=None)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--num_workers", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--save_step", type=int, default=None)
parser.add_argument("--max_text_length", type=int, default=200)
parser.add_argument("--max_audio_length", type=int, default=255995)

args = parser.parse_args()
set_config_from_cli_args(args)

# Training function
def train_gpt(metadatas, is_xtts_v2):
    # Initialize audio config
    audio_config = XttsAudioConfig(
        sample_rate=SAMPLE_RATE,
        dvae_sample_rate=DVAE_SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE
    )
    
    # Initialize model arguments
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        max_wav_length=max_audio_length,
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=8194,
        gpt_start_audio_token=8192,
        gpt_stop_audio_token=8193,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    
    # Initialize config
    config = GPTTrainerConfig(
        model_args=model_args,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACUMM_STEPS,
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        lr=LR,
        optimizer_params={"weight_decay": REG_LOSS_ALPHA},
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5},
        output_path=OUT_PATH,
        save_step=SAVE_STEP,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_step=50,
        log_model_step=100,
        dashboard_logger="tensorboard",
        eval_split_size=0.01,
        print_eval=False,
        binary_align_loss_alpha=0.0,
        kl_loss_alpha=0.0,
        distributed_backend="nccl",
        distributed_url="tcp://localhost:54321",
        mixed_precision=False,
        datasets=metadatas,
        cudnn_enable=True,
        cudnn_deterministic=False,
        cudnn_benchmark=True,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
    )
    
    # Initialize model
    model = GPTTrainer.init_from_config(config)
    
    # **FIX: Initialize tokenizer if not loaded**
    if model.xtts.tokenizer is None:
        print(f" | > Initializing tokenizer from: {TOKENIZER_FILE}")
        model.xtts.tokenizer = VoiceBpeTokenizer(vocab_file=TOKENIZER_FILE)
    
    # Load datasets
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    # **FIX: Safe unknown token check**
    try:
        # Get all transcripts
        all_transcripts = " ".join([sample["text"] for sample in train_samples + eval_samples])
        
        # Check for unknown tokens only if tokenizer is properly initialized
        if model.xtts.tokenizer and hasattr(model.xtts.tokenizer, 'tokenizer') and model.xtts.tokenizer.tokenizer:
            # Get tokenizer vocabulary
            vocab_tokens = set()
            if hasattr(model.xtts.tokenizer.tokenizer, 'get_vocab'):
                vocab = model.xtts.tokenizer.tokenizer.get_vocab()
                for token in vocab.keys():
                    # Extract characters from tokens (remove special markers)
                    clean_token = token.replace('[SPACE]', ' ').strip('[]')
                    vocab_tokens.update(clean_token)
            
            # Find unknown characters
            unknown_chars = sorted(set([char for char in all_transcripts if char not in vocab_tokens]))
            
            if unknown_chars:
                print(f" | > Found {len(unknown_chars)} unknown characters: {unknown_chars[:50]}...")
                print(" | > Warning: These characters may not be properly synthesized")
        else:
            print(" | > Tokenizer vocabulary check skipped (tokenizer not fully initialized)")
    except Exception as e:
        print(f" | > Warning: Could not check for unknown tokens: {e}")
        print(" | > Continuing with training...")
    
    # Initialize trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_clip=1000.0
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Start training
    trainer.fit()
    
    # Get the best model path
    trainer_out_path = trainer.output_path
    
    # Handle files and cleanup
    trainer_tensorboard_dir = os.path.join(trainer_out_path, "tensorboard")
    
    # Copy original model files to output
    os.makedirs(os.path.join(trainer_out_path, "XTTS_v2.0_original_model_files"), exist_ok=True)
    os.system(f"cp -r {XTTS_V2_DIR}/* {os.path.join(trainer_out_path, 'XTTS_v2.0_original_model_files')}")
    
    # Clear memory
    del model, trainer, train_samples, eval_samples
    gc.collect()
    torch.cuda.empty_cache()
    
    return trainer_out_path

# Main execution
if __name__ == "__main__":
    if not DATASETS_CONFIG_LIST:
        raise ValueError("No datasets configured. Use --metadatas argument")
    
    print(f" | > Training with {len(DATASETS_CONFIG_LIST)} dataset(s)")
    print(f" | > Batch size: {BATCH_SIZE}, Grad accumulation: {GRAD_ACUMM_STEPS}")
    print(f" | > Max text length: {max_text_length}, Max audio length: {max_audio_length}")
    
    trainer_out_path = train_gpt(
        metadatas=DATASETS_CONFIG_LIST,
        is_xtts_v2=True
    )
    
    print(f" | > Training completed. Output: {trainer_out_path}")
