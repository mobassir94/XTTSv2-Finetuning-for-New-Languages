import logging
import os
import re
import textwrap
from functools import cached_property

import torch
from num2words import num2words
from tokenizers import Tokenizer

from TTS.tts.layers.xtts.zh_num2words import TextNorm as zh_num2words
from TTS.tts.utils.text.cleaners import collapse_whitespace, lowercase

logger = logging.getLogger(__name__)

def get_spacy_lang(lang):
    try:
        from spacy.lang.ar import Arabic
        from spacy.lang.en import English
        from spacy.lang.es import Spanish
        from spacy.lang.hi import Hindi
        from spacy.lang.bn import Bengali
        from spacy.lang.ja import Japanese
        from spacy.lang.zh import Chinese
        import spacy
    except ImportError as e:
        raise ImportError("enable_text_splitting=True requires Spacy: pip install spacy[ja]") from e
    
    """Return Spacy language used for sentence splitting."""
    if lang == "zh":
        return Chinese()
    elif lang == "ja":
        return Japanese()
    elif lang == "ar":
        return Arabic()
    elif lang == "es":
        return Spanish()
    elif lang == "hi":
        return Hindi()
    elif lang == "bn":
        return Bengali()
    # else:
    #     # For most languages, English does the job
    #     return English()

def split_sentence(text, lang, text_split_length=250):
    """Preprocess the input text"""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        nlp = get_spacy_lang(lang)
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        for sentence in doc.sents:
            if len(text_splits[-1]) + len(str(sentence)) <= text_split_length:
                text_splits[-1] += " " + str(sentence)
                text_splits[-1] = text_splits[-1].lstrip()
            elif len(str(sentence)) > text_split_length:
                for line in textwrap.wrap(
                    str(sentence),
                    width=text_split_length,
                    drop_whitespace=True,
                    break_on_hyphens=False,
                    tabsize=1,
                ):
                    text_splits.append(str(line))
            else:
                text_splits.append(str(sentence))

        if len(text_splits) > 1:
            if text_splits[0] == "":
                del text_splits[0]
    else:
        text_splits = [text.lstrip()]

    return text_splits


# Add Bangla-specific Abbreviations
_abbreviations = {
    "en": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
    "bn": [
        (re.compile(r"\bমি\.", re.IGNORECASE), "মিঃ"),
        (re.compile(r"\bড\.", re.IGNORECASE), "ডাক্তার"),
        (re.compile(r"\bশ্রী\.", re.IGNORECASE), "শ্র"),
        (re.compile(r"\bমা\.", re.IGNORECASE), "মাতৃ"),
    ]
}


# Add Bangla symbols
_symbols_multilingual = {
    "en": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " hash "),
            ("$", " dollar "),
            ("£", " pound "),
            ("°", " degree "),
        ]
    ],
    "bn": [
        (re.compile(r"&", re.IGNORECASE), "এবং"),
        (re.compile(r"@", re.IGNORECASE), "এট দ্য রেট"),
        (re.compile(r"%", re.IGNORECASE), "শতাংশ"),
        (re.compile(r"#", re.IGNORECASE), "হ্যাশ"),
        (re.compile(r"\$", re.IGNORECASE), "টাকা"),
        (re.compile(r"£", re.IGNORECASE), "পাউন্ড"),
        (re.compile(r"°", re.IGNORECASE), "ডিগ্রি"),
    ]
}

def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


# Add Bangla Number Expansion
def bn_num2words(text):
    bangla_numbers = {
        "1": "এক", "2": "দুই", "3": "তিন", "4": "চার", "5": "পাঁচ",
        "6": "ছয়", "7": "সাত", "8": "আট", "9": "নয়", "10": "দশ"
    }
    for num, word in bangla_numbers.items():
        text = text.replace(num, word)
    return text

def expand_numbers_multilingual(text, lang="en"):
    if lang == "bn":
        text = bn_num2words(text)
    else:
        text = re.sub(r"\b\d+\b", lambda m: num2words(m.group(0), lang=lang), text)
    return text


# Integrating Bangla into the multilingual cleaner function
def multilingual_cleaners(text, lang):
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i")
        text = text.replace("Ö", "ö")
        text = text.replace("Ü", "ü")
    text = lowercase(text)
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang=lang)
    text = collapse_whitespace(text)
    return text


def chinese_transliterate(text):
    try:
        import pypinyin
    except ImportError as e:
        raise ImportError("Chinese requires: pypinyin") from e
    return "".join(
        [p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)]
    )


def japanese_cleaners(text, katsu):
    text = katsu.romaji(text)
    text = lowercase(text)
    return text


def korean_transliterate(text):
    try:
        from hangul_romanize import Transliter
        from hangul_romanize.rule import academic
    except ImportError as e:
        raise ImportError("Korean requires: hangul_romanize") from e
    r = Transliter(academic)
    return r.translit(text)


DEFAULT_VOCAB_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/tokenizer.json")


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None):
        self.tokenizer = None
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)
        self.char_limits = {
            "en": 250,
            "de": 253,
            "fr": 273,
            "es": 239,
            "it": 213,
            "pt": 203,
            "pl": 224,
            "zh": 82,
            "ar": 166,
            "cs": 186,
            "ru": 182,
            "nl": 251,
            "tr": 226,
            "ja": 71,
            "hu": 224,
            "ko": 95,
            "hi": 300,
            "bn": 300,  # Set a character limit for Bangla
        }

    @cached_property
    def katsu(self):
        import cutlet
        return cutlet.Cutlet()

    def check_input_length(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        limit = self.char_limits.get(lang, 250)
        if len(txt) > limit:
            logger.warning(
                "The text length exceeds the character limit of %d for language '%s', this might cause truncated audio.",
                limit,
                lang,
            )

    def preprocess_text(self, txt, lang):
        if lang in {"ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko", "bn"}:
            txt = multilingual_cleaners(txt, lang)
            if lang == "zh":
                txt = chinese_transliterate(txt)
            if lang == "ko":
                txt = korean_transliterate(txt)
        elif lang == "ja":
            txt = japanese_cleaners(txt, self.katsu)
        else:
            raise NotImplementedError(f"Language '{lang}' is not supported.")
        return txt

    def encode(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        self.check_input_length(txt, lang)
        txt = self.preprocess_text(txt, lang)
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1

