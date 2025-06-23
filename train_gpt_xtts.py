import os
import gc
import json
import logging
from collections import Counter
from typing import List, Dict, Any

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

import argparse

# Configure logging for encoding and token analysis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_text_encoding(text: str, sample_info: str = "") -> bool:
    """
    Validate that text can be properly encoded/decoded in UTF-8.
    
    Args:
        text: The text string to validate
        sample_info: Additional info about the sample for logging (e.g., file path)
    
    Returns:
        bool: True if encoding is valid, False otherwise
    """
    try:
        # Test UTF-8 encoding/decoding round trip
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        
        # Check for null bytes or other problematic characters
        if '\x00' in text:
            logger.warning(f"Null byte found in text {sample_info}: {repr(text[:50])}")
            return False
            
        # Check for unusual control characters (except common ones like newlines)
        control_chars = [c for c in text if ord(c) < 32 and c not in '\t\n\r']
        if control_chars:
            logger.warning(f"Control characters found in text {sample_info}: {repr(control_chars)}")
            return False
            
        return True
        
    except UnicodeError as e:
        logger.error(f"Encoding error in text {sample_info}: {e}")
        logger.error(f"Problematic text: {repr(text[:100])}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating text {sample_info}: {e}")
        return False

def load_tokenizer_vocab(tokenizer_file: str) -> Dict[str, int]:
    """
    Load the vocabulary from the XTTS tokenizer file.
    
    Args:
        tokenizer_file: Path to the vocab.json file
        
    Returns:
        Dictionary mapping tokens to their IDs
    """
    print("--------loading tokenizer from ",tokenizer_file)
    try:
        with open(tokenizer_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        logger.info(f"Loaded vocabulary with {len(vocab)} tokens")
        return vocab
    except Exception as e:
        logger.error(f"Error loading tokenizer vocabulary: {e}")
        return {}

def analyze_unknown_tokens(texts: List[str], vocab: Dict[str, int], 
                         language: str = "unknown") -> Dict[str, Any]:
    """
    Analyze texts for unknown tokens and provide statistics.
    
    Args:
        texts: List of text strings to analyze
        vocab: Vocabulary dictionary from tokenizer
        language: Language code for logging context
        
    Returns:
        Dictionary with analysis results
    """
    if not vocab:
        logger.warning("No vocabulary provided, skipping unknown token analysis")
        return {"total_texts": len(texts), "analysis_skipped": True}
    
    unknown_tokens = Counter()
    total_tokens = 0
    texts_with_unknowns = 0
    
    # Simple tokenization (split by spaces and punctuation)
    # Note: This is a simplified approach - XTTS may use more sophisticated tokenization
    import re
    
    for i, text in enumerate(texts):
        # Basic tokenization - split on whitespace and separate punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        total_tokens += len(tokens)
        
        text_has_unknowns = False
        for token in tokens:
            if token not in vocab:
                unknown_tokens[token] += 1
                text_has_unknowns = True
        
        if text_has_unknowns:
            texts_with_unknowns += 1
            
        # Log progress for large datasets
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(texts)} texts for unknown token analysis")
    
    # Calculate statistics
    unknown_token_ratio = len(unknown_tokens) / max(total_tokens, 1)
    
    results = {
        "total_texts": len(texts),
        "total_tokens": total_tokens,
        "unique_unknown_tokens": len(unknown_tokens),
        "total_unknown_occurrences": sum(unknown_tokens.values()),
        "unknown_token_ratio": unknown_token_ratio,
        "texts_with_unknowns": texts_with_unknowns,
        "most_common_unknowns": unknown_tokens.most_common(20),
        "language": language
    }
    
    return results

def log_token_analysis_results(results: Dict[str, Any]):
    """Log the results of unknown token analysis in a readable format."""
    if results.get("analysis_skipped"):
        logger.warning("Unknown token analysis was skipped")
        return
    
    logger.info("="*60)
    logger.info(f"UNKNOWN TOKEN ANALYSIS RESULTS - Language: {results['language']}")
    logger.info("="*60)
    logger.info(f"Total texts analyzed: {results['total_texts']}")
    logger.info(f"Total tokens: {results['total_tokens']}")
    logger.info(f"Unique unknown tokens: {results['unique_unknown_tokens']}")
    logger.info(f"Total unknown token occurrences: {results['total_unknown_occurrences']}")
    logger.info(f"Unknown token ratio: {results['unknown_token_ratio']:.4f} ({results['unknown_token_ratio']*100:.2f}%)")
    logger.info(f"Texts containing unknowns: {results['texts_with_unknowns']}/{results['total_texts']}")
    
    if results['most_common_unknowns']:
        logger.info("\nMost common unknown tokens:")
        for token, count in results['most_common_unknowns']:
            logger.info(f"  '{token}': {count} occurrences")
    
    # Provide recommendations based on results
    if results['unknown_token_ratio'] > 0.05:  # More than 5% unknown tokens
        logger.warning("⚠️  High ratio of unknown tokens detected!")
        logger.warning("Consider:")
        logger.warning("- Cleaning your text data")
        logger.warning("- Using a different tokenizer")
        logger.warning("- Expanding the vocabulary")
    elif results['unknown_token_ratio'] > 0.01:  # More than 1% unknown tokens
        logger.info("ℹ️  Moderate unknown token ratio - monitor training performance")
    else:
        logger.info("✅ Low unknown token ratio - good text compatibility")
    
    logger.info("="*60)

def validate_and_analyze_training_data(train_samples: List[Dict], eval_samples: List[Dict], 
                                     tokenizer_file: str, language: str = "unknown"):
    """
    Comprehensive validation and analysis of training data.
    
    Args:
        train_samples: Training dataset samples
        eval_samples: Evaluation dataset samples  
        tokenizer_file: Path to tokenizer vocabulary file
        language: Language code for context
    """
    logger.info(f"Starting data validation and analysis for {language}")
    
    # Combine all samples for analysis
    all_samples = train_samples + eval_samples
    total_samples = len(all_samples)
    
    # Encoding validation
    logger.info("Performing encoding validation...")
    encoding_failures = 0
    valid_texts = []
    
    for i, sample in enumerate(all_samples):
        text = sample.get('text', '')
        audio_file = sample.get('audio_file', 'unknown')
        
        if validate_text_encoding(text, f"from {audio_file}"):
            valid_texts.append(text)
        else:
            encoding_failures += 1
            logger.error(f"Encoding failure in sample {i}: {audio_file}")
    
    logger.info(f"Encoding validation complete: {len(valid_texts)}/{total_samples} samples valid")
    
    if encoding_failures > 0:
        logger.warning(f"⚠️  {encoding_failures} samples failed encoding validation!")
        logger.warning("These samples may cause training issues")
    
    # Unknown token analysis
    if os.path.exists(tokenizer_file) and valid_texts:
        logger.info("Loading tokenizer and analyzing unknown tokens...")
        vocab = load_tokenizer_vocab(tokenizer_file)
        
        if vocab:
            # Analyze training texts
            train_texts = [s['text'] for s in train_samples if validate_text_encoding(s.get('text', ''))]
            eval_texts = [s['text'] for s in eval_samples if validate_text_encoding(s.get('text', ''))]
            
            logger.info("Analyzing training set...")
            train_results = analyze_unknown_tokens(train_texts, vocab, f"{language}_train")
            log_token_analysis_results(train_results)
            
            if eval_texts:
                logger.info("Analyzing evaluation set...")
                eval_results = analyze_unknown_tokens(eval_texts, vocab, f"{language}_eval")
                log_token_analysis_results(eval_results)
    else:
        logger.warning("Skipping unknown token analysis - tokenizer file not found or no valid texts")

def create_xtts_trainer_parser():
    parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to pretrained + checkpoint model")
    parser.add_argument("--metadatas", nargs='+', type=str, required=True,
                        help="train_csv_path,eval_csv_path,language")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini batch size")
    parser.add_argument("--grad_acumm", type=int, default=1,
                        help="Grad accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=255995,
                        help="Max audio length")
    parser.add_argument("--max_text_length", type=int, default=200,
                        help="Max text length")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Save step")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip encoding and token validation (faster but less safe)")

    return parser

def train_gpt(metadatas, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, 
              max_text_length, lr, weight_decay, save_step, skip_validation=False):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    # OUT_PATH = os.path.join(output_path, "run", "training")
    OUT_PATH = output_path

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps

    # Define here the dataset that you want to use for the fine-tuning on.
    DATASETS_CONFIG_LIST = []
    for metadata in metadatas:
        train_csv, eval_csv, language = metadata.split(",")
        print(train_csv, eval_csv, language)

        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )

        DATASETS_CONFIG_LIST.append(config_dataset)

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "/vocab.json"
    XTTS_CHECKPOINT_LINK = "/model.pth"
    XTTS_CONFIG_LINK = "/config.json"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files(
            [XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files(
            [XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=11025,  # 0.5 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config

    config = GPTTrainerConfig()

    config.load_json(XTTS_CONFIG_FILE)

    config.epochs = num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = """
        GPT XTTS training
        """,
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay}
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Perform comprehensive data validation and analysis
    if not skip_validation:
        for i, metadata in enumerate(metadatas):
            _, _, language = metadata.split(",")
            
            # Get samples for this specific language/dataset
            # Note: This assumes each metadata corresponds to samples in order
            # You may need to adjust this logic based on how your datasets are structured
            dataset_train_samples = train_samples  # Adjust as needed for multi-dataset scenarios
            dataset_eval_samples = eval_samples    # Adjust as needed for multi-dataset scenarios
            
            validate_and_analyze_training_data(
                dataset_train_samples, 
                dataset_eval_samples, 
                TOKENIZER_FILE, 
                language
            )
    else:
        logger.info("Skipping data validation (--skip_validation flag set)")

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS
        ),
        config,
        output_path=os.path.join(output_path, "run", "training"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx =  samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return trainer_out_path

if __name__ == "__main__":
    parser = create_xtts_trainer_parser()
    args = parser.parse_args()

    trainer_out_path = train_gpt(
        metadatas=args.metadatas,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_acumm,
        weight_decay=args.weight_decay,
        lr=args.lr,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        save_step=args.save_step,
        skip_validation=args.skip_validation
    )

    print(f"Checkpoint saved in dir: {trainer_out_path}")
