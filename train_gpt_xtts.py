from tokenizers import Tokenizer
from collections import Counter

def load_tokenizer_vocab(tokenizer_file: str):
    """Load XTTS tokenizer properly"""
    print("--------loading tokenizer from ", tokenizer_file)
    try:
        tokenizer = Tokenizer.from_file(tokenizer_file)
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"Loaded vocabulary with {vocab_size} tokens")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return None

def analyze_unknown_tokens(texts: List[str], tokenizer, language: str = "unknown") -> Dict[str, Any]:
    """Analyze texts using actual XTTS tokenizer"""
    if not tokenizer:
        logger.warning("No tokenizer provided, skipping analysis")
        return {"total_texts": len(texts), "analysis_skipped": True}
    
    unknown_chars = Counter()
    total_tokens = 0
    total_unk_tokens = 0
    texts_with_unknowns = 0
    
    for i, text in enumerate(texts):
        try:
            # Use actual tokenizer to encode
            encoded = tokenizer.encode(text)
            tokens = encoded.tokens
            total_tokens += len(tokens)
            
            # Count UNK tokens
            unk_count = tokens.count('[UNK]')
            total_unk_tokens += unk_count
            
            if unk_count > 0:
                texts_with_unknowns += 1
                
                # Find which characters cause UNK tokens
                for char in text:
                    char_encoded = tokenizer.encode(char)
                    if '[UNK]' in char_encoded.tokens:
                        unknown_chars[char] += 1
                        
        except Exception as e:
            logger.warning(f"Error encoding text {i}: {e}")
            continue
            
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(texts)} texts")
    
    unknown_token_ratio = total_unk_tokens / max(total_tokens, 1)
    
    return {
        "total_texts": len(texts),
        "total_tokens": total_tokens,
        "total_unk_tokens": total_unk_tokens,
        "unknown_token_ratio": unknown_token_ratio,
        "texts_with_unknowns": texts_with_unknowns,
        "most_common_unknowns": unknown_chars.most_common(20),
        "language": language
    }

def log_token_analysis_results(results: Dict[str, Any]):
    """Log analysis results"""
    if results.get("analysis_skipped"):
        logger.warning("Unknown token analysis was skipped")
        return
    
    logger.info("="*60)
    logger.info(f"XTTS TOKEN ANALYSIS - Language: {results['language']}")
    logger.info("="*60)
    logger.info(f"Total texts: {results['total_texts']}")
    logger.info(f"Total tokens: {results['total_tokens']}")
    logger.info(f"UNK tokens: {results['total_unk_tokens']}")
    logger.info(f"UNK ratio: {results['unknown_token_ratio']:.4f} ({results['unknown_token_ratio']*100:.2f}%)")
    logger.info(f"Texts with UNK: {results['texts_with_unknowns']}/{results['total_texts']}")
    
    if results['most_common_unknowns']:
        logger.info("\nMost common unknown characters:")
        for char, count in results['most_common_unknowns']:
            logger.info(f"  '{char}': {count} occurrences")
    
    if results['unknown_token_ratio'] > 0.05:
        logger.warning("⚠️  High UNK token ratio!")
    elif results['unknown_token_ratio'] > 0.01:
        logger.info("ℹ️  Moderate UNK tokens")
    else:
        logger.info("✅ Low UNK token ratio")
    
    logger.info("="*60)

def validate_and_analyze_training_data(train_samples: List[Dict], eval_samples: List[Dict], 
                                     tokenizer_file: str, language: str = "unknown"):
    """Fixed validation using actual XTTS tokenizer"""
    logger.info(f"Starting XTTS data analysis for {language}")
    
    # Load tokenizer properly
    tokenizer = load_tokenizer_vocab(tokenizer_file)
    if not tokenizer:
        logger.error("Failed to load tokenizer")
        return
    
    # Test tokenizer with Hindi
    test_text = "यह हिंदी का परीक्षण है"
    encoded = tokenizer.encode(test_text)
    unk_count = encoded.tokens.count('[UNK]')
    logger.info(f"Tokenizer test - UNK tokens in '{test_text}': {unk_count}")
    if unk_count > 0:
        logger.warning("⚠️  Hindi characters still producing UNK tokens!")
    
    # Analyze datasets
    train_texts = [s['text'] for s in train_samples]
    eval_texts = [s['text'] for s in eval_samples]
    
    logger.info("Analyzing training set...")
    train_results = analyze_unknown_tokens(train_texts, tokenizer, f"{language}_train")
    log_token_analysis_results(train_results)
    
    if eval_texts:
        logger.info("Analyzing evaluation set...")
        eval_results = analyze_unknown_tokens(eval_texts, tokenizer, f"{language}_eval")
        log_token_analysis_results(eval_results)
