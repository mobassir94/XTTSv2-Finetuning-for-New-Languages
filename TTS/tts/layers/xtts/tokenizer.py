import os
import re
import textwrap
from functools import cached_property

import pypinyin
import torch
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from num2words import num2words
from tokenizers import Tokenizer

from TTS.tts.layers.xtts.zh_num2words import TextNorm as zh_num2words

# Try to import IndicNLP modules
try:
    from indicnlp.tokenize import indic_tokenize
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import sentence_tokenize
    INDICNLP_AVAILABLE = True
except ImportError:
    INDICNLP_AVAILABLE = False
    print("[!] Warning: IndicNLP not available. Install it for better Indic language support.")

# Fallback imports for non-Indic languages
try:
    from spacy.lang.ar import Arabic
    from spacy.lang.en import English
    from spacy.lang.es import Spanish
    from spacy.lang.ja import Japanese
    from spacy.lang.zh import Chinese
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("[!] Warning: spaCy not available for non-Indic languages.")


def get_sentence_tokenizer(lang):
    """Get appropriate sentence tokenizer for the language"""
    if lang in ["hi", "bn", "ta", "te", "ml", "kn", "gu", "mr", "pa", "or", "as", "ne"] and INDICNLP_AVAILABLE:
        # Use IndicNLP for Indic languages
        return lambda text: sentence_tokenize.sentence_split(text, lang=lang)
    elif SPACY_AVAILABLE:
        # Use spaCy for other languages
        if lang == "zh":
            nlp = Chinese()
        elif lang == "ja":
            nlp = Japanese()
        elif lang == "ar":
            nlp = Arabic()
        elif lang == "es":
            nlp = Spanish()
        else:
            nlp = English()
        nlp.add_pipe("sentencizer")
        return lambda text: [str(sent) for sent in nlp(text).sents]
    else:
        # Simple fallback sentence splitter
        return lambda text: re.split(r'[.!?।॥।]\s*', text)


def split_sentence(text, lang, text_split_length=250):
    """Preprocess the input text"""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        
        # Get appropriate tokenizer
        tokenizer = get_sentence_tokenizer(lang)
        sentences = tokenizer(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(text_splits[-1]) + len(sentence) <= text_split_length:
                # if the last sentence + the current sentence is less than the text_split_length
                # then add the current sentence to the last sentence
                text_splits[-1] += " " + sentence
                text_splits[-1] = text_splits[-1].lstrip()
            elif len(sentence) > text_split_length:
                # if the current sentence is greater than the text_split_length
                for line in textwrap.wrap(
                    sentence,
                    width=text_split_length,
                    drop_whitespace=True,
                    break_on_hyphens=False,
                    tabsize=1,
                ):
                    text_splits.append(str(line))
            else:
                text_splits.append(sentence)

        if len(text_splits) > 1:
            if text_splits[0] == "":
                del text_splits[0]
    else:
        text_splits = [text.lstrip()]

    return text_splits


_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
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
    "es": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"),
            ("sr", "señor"),
            ("dr", "doctor"),
            ("dra", "doctora"),
            ("st", "santo"),
            ("co", "compañía"),
            ("jr", "junior"),
            ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"),
            ("mr", "monsieur"),
            ("dr", "docteur"),
            ("st", "saint"),
            ("co", "compagnie"),
            ("jr", "junior"),
            ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"),
            ("dr", "doktor"),
            ("st", "sankt"),
            ("co", "firma"),
            ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"),
            ("sr", "senhor"),
            ("dr", "doutor"),
            ("dra", "doutora"),
            ("st", "santo"),
            ("co", "companhia"),
            ("jr", "júnior"),
            ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # ("sig.ra", "signora"),
            ("sig", "signore"),
            ("dr", "dottore"),
            ("st", "santo"),
            ("co", "compagnia"),
            ("jr", "junior"),
            ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"),
            ("m", "pan"),
            ("dr", "doktor"),
            ("sw", "święty"),
            ("jr", "junior"),
        ]
    ],
    "ar": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # There are not many common abbreviations in Arabic as in English.
        ]
    ],
    "zh": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Chinese doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "cs": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("ing", "inženýr"),  # engineer
            ("p", "pan"),  # Could also map to pani for woman but no easy way to do it
            # Other abbreviations would be specialized and not as common.
        ]
    ],
    "ru": [
        (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"),  # Mrs.
            ("г-н", "господин"),  # Mr.
            ("д-р", "доктор"),  # doctor
            # Other abbreviations are less common or specialized.
        ]
    ],
    "nl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"),  # Mr.
            ("mevr", "mevrouw"),  # Mrs.
            ("dr", "dokter"),  # doctor
            ("jhr", "jonkheer"),  # young lord or nobleman
            # Dutch uses more abbreviations, but these are the most common ones.
        ]
    ],
    "tr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("b", "bay"),  # Mr.
            ("byk", "büyük"),  # büyük
            ("dr", "doktor"),  # doctor
            # Add other Turkish abbreviations here if needed.
        ]
    ],
    "hu": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("b", "bácsi"),  # Mr.
            ("nőv", "nővér"),  # nurse
            # Add other Hungarian abbreviations here if needed.
        ]
    ],
    "ko": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Korean doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "hi": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("श्री", "श्री"),  # Mr.
            ("श्रीमती", "श्रीमती"),  # Mrs.
            ("डॉ", "डॉक्टर"),  # Doctor
            ("प्रो", "प्रोफेसर"),  # Professor
            ("कं", "कंपनी"),  # Company
            ("लि", "लिमिटेड"),  # Limited
            ("सं", "संस्था"),  # Institution
            ("प्र", "प्रधान"),  # Chief/Principal
            ("उ", "उत्तर"),  # North
            ("द", "दक्षिण"),  # South
            ("पू", "पूर्व"),  # East
            ("प", "पश्चिम"),  # West
        ]
    ],
    "bn": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("শ্রী", "শ্রী"),  # Mr.
            ("শ্রীমতী", "শ্রীমতী"),  # Mrs.
            ("ড", "ডাক্তার"),  # Doctor
            ("প্রফ", "প্রফেসর"),  # Professor
            ("কো", "কোম্পানি"),  # Company
            ("লি", "লিমিটেড"),  # Limited
        ]
    ],
}


def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


_symbols_multilingual = {
    "en": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
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
    "es": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " y "),
            ("@", " arroba "),
            ("%", " por ciento "),
            ("#", " numeral "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " grados "),
        ]
    ],
    "fr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " et "),
            ("@", " arobase "),
            ("%", " pour cent "),
            ("#", " dièse "),
            ("$", " dollar "),
            ("£", " livre "),
            ("°", " degrés "),
        ]
    ],
    "de": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " und "),
            ("@", " at "),
            ("%", " prozent "),
            ("#", " raute "),
            ("$", " dollar "),
            ("£", " pfund "),
            ("°", " grad "),
        ]
    ],
    "pt": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " arroba "),
            ("%", " por cento "),
            ("#", " cardinal "),
            ("$", " dólar "),
            ("£", " libra "),
            ("°", " graus "),
        ]
    ],
    "it": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " chiocciola "),
            ("%", " per cento "),
            ("#", " cancelletto "),
            ("$", " dollaro "),
            ("£", " sterlina "),
            ("°", " gradi "),
        ]
    ],
    "pl": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " i "),
            ("@", " małpa "),
            ("%", " procent "),
            ("#", " krzyżyk "),
            ("$", " dolar "),
            ("£", " funt "),
            ("°", " stopnie "),
        ]
    ],
    "ar": [
        # Arabic
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " و "),
            ("@", " على "),
            ("%", " في المئة "),
            ("#", " رقم "),
            ("$", " دولار "),
            ("£", " جنيه "),
            ("°", " درجة "),
        ]
    ],
    "zh": [
        # Chinese
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 和 "),
            ("@", " 在 "),
            ("%", " 百分之 "),
            ("#", " 号 "),
            ("$", " 美元 "),
            ("£", " 英镑 "),
            ("°", " 度 "),
        ]
    ],
    "cs": [
        # Czech
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " a "),
            ("@", " na "),
            ("%", " procento "),
            ("#", " křížek "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " stupně "),
        ]
    ],
    "ru": [
        # Russian
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " и "),
            ("@", " собака "),
            ("%", " процентов "),
            ("#", " номер "),
            ("$", " доллар "),
            ("£", " фунт "),
            ("°", " градус "),
        ]
    ],
    "nl": [
        # Dutch
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " en "),
            ("@", " bij "),
            ("%", " procent "),
            ("#", " hekje "),
            ("$", " dollar "),
            ("£", " pond "),
            ("°", " graden "),
        ]
    ],
    "tr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " ve "),
            ("@", " at "),
            ("%", " yüzde "),
            ("#", " diyez "),
            ("$", " dolar "),
            ("£", " sterlin "),
            ("°", " derece "),
        ]
    ],
    "hu": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " és "),
            ("@", " kukac "),
            ("%", " százalék "),
            ("#", " kettőskereszt "),
            ("$", " dollár "),
            ("£", " font "),
            ("°", " fok "),
        ]
    ],
    "ko": [
        # Korean
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 그리고 "),
            ("@", " 에 "),
            ("%", " 퍼센트 "),
            ("#", " 번호 "),
            ("$", " 달러 "),
            ("£", " 파운드 "),
            ("°", " 도 "),
        ]
    ],
    "hi": [
        # Hindi
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " और "),
            ("@", " पर "),
            ("%", " प्रतिशत "),
            ("#", " नंबर "),
            ("$", " डॉलर "),
            ("£", " पाउंड "),
            ("°", " डिग्री "),
            ("₹", " रुपये "),
            ("€", " यूरो "),
            ("+", " प्लस "),
            ("=", " बराबर "),
            ("÷", " भाग "),
            ("×", " गुणा "),
        ]
    ],
    "bn": [
        # Bangla
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " এবং "),
            ("@", " এ "),
            ("%", " শতাংশ "),
            ("#", " নম্বর "),
            ("$", " ডলার "),
            ("£", " পাউন্ড "),
            ("°", " ডিগ্রি "),
            ("৳", " টাকা "),
        ]
    ],
}


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
    "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
    "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
    "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
    "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
    "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
    "cs": re.compile(r"([0-9]+)\.(?=\s|$)"),  # In Czech, a dot is often used after the number to indicate ordinals.
    "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
    "nl": re.compile(r"([0-9]+)(de|ste|e)"),
    "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
    "hu": re.compile(r"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"),
    "ko": re.compile(r"([0-9]+)(번째|번|차|째)"),
    "hi": re.compile(r"([0-9]+)(वां|वीं|वें|था|थी|थे)"),
    "bn": re.compile(r"([0-9]+)(তম|য়|ম|টি|টা)"),
}

# Hindi specific number patterns
_hindi_number_re = re.compile(r"[०-९]+")
_hindi_digit_map = str.maketrans("०१२३४५६७८९", "0123456789")

# Bangla specific number patterns
_bangla_number_re = re.compile(r"[০-৯]+")
_bangla_digit_map = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
    "INR": re.compile(r"((₹[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+₹))"),
    "BDT": re.compile(r"((৳[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+৳))"),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")

# Hindi number words mapping
_hindi_ones = ["", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ"]
_hindi_tens = ["", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"]
_hindi_teens = ["दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस"]


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text


def _expand_decimal_point(m, lang="en"):
    amount = m.group(1).replace(",", ".")
    if lang == "hi":
        parts = amount.split(".")
        return hindi_number_to_words(int(parts[0])) + " दशमलव " + " ".join([hindi_number_to_words(int(d)) for d in parts[1]])
    else:
        return num2words(float(amount), lang=lang if lang != "cs" else "cz")


def _expand_currency(m, lang="en", currency="USD"):
    amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
    
    if lang == "hi" and currency == "INR":
        if amount.is_integer():
            return hindi_number_to_words(int(amount)) + " रुपये"
        else:
            rupees = int(amount)
            paise = int((amount - rupees) * 100)
            return hindi_number_to_words(rupees) + " रुपये " + hindi_number_to_words(paise) + " पैसे"
    else:
        full_amount = num2words(amount, to="currency", currency=currency, lang=lang if lang != "cs" else "cz")
        
        and_equivalents = {
            "en": ", ",
            "es": " con ",
            "fr": " et ",
            "de": " und ",
            "pt": " e ",
            "it": " e ",
            "pl": ", ",
            "cs": ", ",
            "ru": ", ",
            "nl": ", ",
            "ar": ", ",
            "tr": ", ",
            "hu": ", ",
            "ko": ", ",
            "hi": ", ",
            "bn": ", ",
        }

        if amount.is_integer():
            last_and = full_amount.rfind(and_equivalents[lang])
            if last_and != -1:
                full_amount = full_amount[:last_and]

        return full_amount


def _expand_ordinal(m, lang="en"):
    if lang == "hi":
        return hindi_number_to_words(int(m.group(1))) + m.group(2)
    else:
        return num2words(int(m.group(1)), ordinal=True, lang=lang if lang != "cs" else "cz")


def _expand_number(m, lang="en"):
    if lang == "hi":
        return hindi_number_to_words(int(m.group(0)))
    else:
        return num2words(int(m.group(0)), lang=lang if lang != "cs" else "cz")


def hindi_number_to_words(num):
    """Convert number to Hindi words"""
    if num == 0:
        return "शून्य"
    
    if num < 0:
        return "ऋण " + hindi_number_to_words(-num)
    
    if num < 10:
        return _hindi_ones[num]
    elif num < 20:
        return _hindi_teens[num - 10]
    elif num < 100:
        tens = num // 10
        ones = num % 10
        if ones == 0:
            return _hindi_tens[tens]
        elif tens == 1:
            return _hindi_teens[ones]
        else:
            return _hindi_tens[tens] + " " + _hindi_ones[ones] if ones else _hindi_tens[tens]
    elif num < 1000:
        hundreds = num // 100
        remainder = num % 100
        result = _hindi_ones[hundreds] + " सौ"
        if remainder:
            result += " " + hindi_number_to_words(remainder)
        return result
    elif num < 100000:  # 1 lakh
        thousands = num // 1000
        remainder = num % 1000
        result = hindi_number_to_words(thousands) + " हज़ार"
        if remainder:
            result += " " + hindi_number_to_words(remainder)
        return result
    elif num < 10000000:  # 1 crore
        lakhs = num // 100000
        remainder = num % 100000
        result = hindi_number_to_words(lakhs) + " लाख"
        if remainder:
            result += " " + hindi_number_to_words(remainder)
        return result
    else:
        crores = num // 10000000
        remainder = num % 10000000
        result = hindi_number_to_words(crores) + " करोड़"
        if remainder:
            result += " " + hindi_number_to_words(remainder)
        return result


def normalize_hindi_numerals(text):
    """Convert Hindi numerals to Arabic numerals"""
    return text.translate(_hindi_digit_map)


def normalize_bangla_numerals(text):
    """Convert Bangla numerals to Arabic numerals"""
    return text.translate(_bangla_digit_map)


def expand_numbers_multilingual(text, lang="en"):
    if lang == "zh":
        text = zh_num2words()(text)
    elif lang == "hi":
        # First convert Hindi numerals to Arabic
        text = normalize_hindi_numerals(text)
        # Handle currency
        text = re.sub(_currency_re["INR"], lambda m: _expand_currency(m, lang, "INR"), text)
        # Handle decimal numbers
        text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
        # Handle ordinals
        text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
        # Handle regular numbers
        text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    elif lang == "bn":
        # First convert Bangla numerals to Arabic
        text = normalize_bangla_numerals(text)
        # Handle currency
        text = re.sub(_currency_re["BDT"], lambda m: _expand_currency(m, lang, "BDT"), text)
        # Handle ordinals
        text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, "en"), text)
        # Handle regular numbers
        text = re.sub(_number_re, lambda m: _expand_number(m, "en"), text)
    else:
        if lang in ["en", "ru"]:
            text = re.sub(_comma_number_re, _remove_commas, text)
        else:
            text = re.sub(_dot_number_re, _remove_dots, text)
        try:
            text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
            text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
            text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
        except:
            pass
        if lang != "tr":
            text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
        text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
        text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def indic_cleaners(text, lang="hi"):
    """Indic language specific text cleaning using IndicNLP"""
    if INDICNLP_AVAILABLE:
        # Initialize normalizer factory
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer(lang)
        
        # Normalize the text
        text = normalizer.normalize(text)
    
    # Language-specific punctuation normalization
    if lang == "hi":
        text = hindi_cleaners(text)
    elif lang == "bn":
        text = bangla_cleaners(text)
    else:
        # Generic Indic cleaning
        text = text.replace("।", ".")  # Convert danda to period
        text = text.replace("॥", ".")  # Convert double danda
    
    return text


def hindi_cleaners(text):
    """Hindi-specific text cleaning"""
    # Normalize Devanagari punctuation
    text = text.replace("।", ".")  # Convert Devanagari full stop
    text = text.replace("॥", ".")  # Convert double danda
    text = text.replace("ऽ", "")    # Remove avagraha
    text = text.replace("॰", ".")   # Convert abbreviation sign
    
    # Handle zero-width characters
    text = text.replace("\u200c", "")  # Remove ZWNJ
    text = text.replace("\u200d", "")  # Remove ZWJ
    
    # Normalize quotes
    text = text.replace(""", '"')
    text = text.replace(""", '"')
    text = text.replace("'", "'")
    text = text.replace("'", "'")
    
    # Handle mixed script (English words in Hindi text)
    text = re.sub(r'([a-zA-Z]+)', r' \1 ', text)  # Add spaces around English words
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def bangla_cleaners(text):
    """Bangla-specific text cleaning"""
    # Normalize Bangla punctuation
    text = text.replace("।", ".")  # Convert Bangla dari
    text = text.replace("॥", ".")  # Convert double dari
    
    # Handle zero-width characters
    text = text.replace("\u200c", "")  # Remove ZWNJ
    text = text.replace("\u200d", "")  # Remove ZWJ
    
    # Handle mixed script (English words in Bangla text)
    text = re.sub(r'([a-zA-Z]+)', r' \1 ', text)  # Add spaces around English words
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


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


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def chinese_transliterate(text):
    return "".join(
        [p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)]
    )


def japanese_cleaners(text, katsu):
    text = katsu.romaji(text)
    text = lowercase(text)
    return text


def korean_transliterate(text):
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
            # Indic languages with higher limits
            "hi": 400,  # Hindi
            "bn": 400,  # Bangla
            "ta": 350,  # Tamil
            "te": 350,  # Telugu
            "mr": 350,  # Marathi
            "gu": 350,  # Gujarati
            "ml": 350,  # Malayalam
            "kn": 350,  # Kannada
            "or": 350,  # Odia
            "pa": 350,  # Punjabi
            "as": 350,  # Assamese
            "ne": 350,  # Nepali
        }

    @cached_property
    def katsu(self):
        import cutlet

        return cutlet.Cutlet()

    def check_input_length(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        limit = self.char_limits.get(lang, 300)
        if len(txt) > limit:
            print(
                f"[!] Warning: The text length exceeds the character limit of {limit} for language '{lang}', this might cause truncated audio."
            )

    def preprocess_text(self, txt, lang):
        if lang in {"ar", "cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"}:
            txt = multilingual_cleaners(txt, lang)
            if lang == "zh":
                txt = chinese_transliterate(txt)
            if lang == "ko":
                txt = korean_transliterate(txt)
        elif lang == "ja":
            txt = japanese_cleaners(txt, self.katsu)
        elif lang in {"hi", "bn", "ta", "te", "mr", "gu", "ml", "kn", "or", "pa", "as", "ne"}:
            # Use IndicNLP for Indic languages
            txt = indic_cleaners(txt, lang)
            txt = multilingual_cleaners(txt, lang)
        else:
            txt = basic_cleaners(txt)
            # print(f"[!] Warning: Preprocess [Language '{lang}'] text is not implemented, use `basic_cleaners` instead.")
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


def test_expand_numbers_multilingual():
    """Test number expansion with focus on Hindi"""
    test_cases = [
        # Hindi test cases
        ("मेरे पास 50 रुपये हैं।", "मेरे पास पचास रुपये हैं।", "hi"),
        ("आज 25 दिसंबर है।", "आज पच्चीस दिसंबर है।", "hi"),
        ("मैं 100 किलोमीटर गया।", "मैं एक सौ किलोमीटर गया।", "hi"),
        ("₹500 का नोट", "पांच सौ रुपये का नोट", "hi"),
        ("₹1000 की किताब", "एक हज़ार रुपये की किताब", "hi"),
        ("₹2500 का सामान", "दो हज़ार पांच सौ रुपये का सामान", "hi"),
        ("यह 1वां टेस्ट है", "यह एकवां टेस्ट है", "hi"),
        ("3था दिन", "तीनथा दिन", "hi"),
        ("21वीं सदी", "इक्कीसवीं सदी", "hi"),
        ("१२३ संख्या", "एक सौ तेईस संख्या", "hi"),  # Hindi numerals
        ("९९९ तक", "नौ सौ निन्यानवे तक", "hi"),  # Hindi numerals
        ("3.5 किलो", "तीन दशमलव पांच किलो", "hi"),
        ("12.75 प्रतिशत", "बारह दशमलव सात पांच प्रतिशत", "hi"),
        # Test with mixed English-Hindi
        ("मुझे 25% discount मिला", "मुझे पच्चीस प्रतिशत discount मिला", "hi"),
        ("Total ₹1500 है", "total एक हज़ार पांच सौ रुपये है", "hi"),
        # Large numbers
        ("100000 लोग", "एक लाख लोग", "hi"),
        ("2500000 की आबादी", "पच्चीस लाख की आबादी", "hi"),
        ("10000000 का बजट", "एक करोड़ का बजट", "hi"),
        # Keep some English test cases for comparison
        ("In 12.5 seconds.", "In twelve point five seconds.", "en"),
        ("This is a 1st test", "This is a first test", "en"),
        ("That will be $20 sir.", "That will be twenty dollars sir.", "en"),
    ]
    
    for a, b, lang in test_cases:
        out = expand_numbers_multilingual(a, lang=lang)
        assert out == b, f"Test failed for '{a}': got '{out}' but expected '{b}'"
    
    print("All number expansion tests passed!")


def test_abbreviations_multilingual():
    """Test abbreviation expansion with focus on Hindi"""
    test_cases = [
        # Hindi test cases
        ("डॉ. शर्मा यहाँ हैं।", "डॉक्टर शर्मा यहाँ हैं।", "hi"),
        ("श्री राम जी", "श्री राम जी", "hi"),
        ("प्रो. वर्मा", "प्रोफेसर वर्मा", "hi"),
        ("कं. लिमिटेड", "कंपनी लिमिटेड", "hi"),
        ("उ. प्रदेश", "उत्तर प्रदेश", "hi"),
        ("द. भारत", "दक्षिण भारत", "hi"),
        ("पू. दिशा", "पूर्व दिशा", "hi"),
        ("प. बंगाल", "पश्चिम बंगाल", "hi"),
        # Keep some English test cases
        ("Hello Mr. Smith.", "Hello mister Smith.", "en"),
        ("Dr. Jones is here.", "doctor Jones is here.", "en"),
    ]

    for a, b, lang in test_cases:
        out = expand_abbreviations_multilingual(a, lang=lang)
        assert out == b, f"Test failed for '{a}': got '{out}' but expected '{b}'"
    
    print("All abbreviation tests passed!")


def test_symbols_multilingual():
    """Test symbol expansion with focus on Hindi"""
    test_cases = [
        # Hindi test cases
        ("मुझे 50% छूट मिली", "मुझे 50 प्रतिशत छूट मिली", "hi"),
        ("₹100 & $50", " रुपये 100 और  डॉलर 50", "hi"),
        ("मेरा email @ gmail.com", "मेरा email पर gmail.com", "hi"),
        ("10°C तापमान", "10 डिग्री c तापमान", "hi"),
        ("A & B कंपनी", "a और b कंपनी", "hi"),
        ("50% + 20%", "50 प्रतिशत प्लस 20 प्रतिशत", "hi"),
        ("100 × 2 = 200", "100 गुणा 2 बराबर 200", "hi"),
        ("10 ÷ 2", "10 भाग 2", "hi"),
        ("€50 का सामान", " यूरो 50 का सामान", "hi"),
        ("Question #5", "question नंबर 5", "hi"),
        # Keep some English test cases
        ("I have 14% battery", "I have 14 percent battery", "en"),
        ("Meet me @ 5pm", "Meet me at 5pm", "en"),
    ]

    for a, b, lang in test_cases:
        out = expand_symbols_multilingual(a, lang=lang)
        assert out == b, f"Test failed for '{a}': got '{out}' but expected '{b}'"
    
    print("All symbol tests passed!")


def test_hindi_cleaners():
    """Test Hindi-specific cleaning functions"""
    test_cases = [
        # Punctuation normalization
        ("यह वाक्य है।", "यह वाक्य है."),
        ("राम॥श्याम", "राम.श्याम"),
        ("क॰ ख॰ ग॰", "क. ख. ग."),
        # Mixed script handling
        ("मैं Python सीख रहा हूं", "मैं Python सीख रहा हूं"),
        ("Hindi और English मिक्स", "Hindi और English मिक्स"),
        # Quote normalization
        (""यह उद्धरण है"", '"यह उद्धरण है"'),
        ("'एक और उदाहरण'", "'एक और उदाहरण'"),
        # Extra spaces
        ("बहुत    ज्यादा    स्पेस", "बहुत ज्यादा स्पेस"),
    ]
    
    for input_text, expected in test_cases:
        output = hindi_cleaners(input_text)
        assert output == expected, f"Test failed for '{input_text}': got '{output}' but expected '{expected}'"
    
    print("All Hindi cleaner tests passed!")


def test_full_preprocessing():
    """Test full preprocessing pipeline for Hindi"""
    tokenizer = VoiceBpeTokenizer()
    
    test_cases = [
        ("डॉ. शर्मा ने कहा कि ₹500 का सामान 25% छूट पर मिलेगा।", "hi"),
        ("आज १५ अगस्त है और हम 75वां स्वतंत्रता दिवस मना रहे हैं।", "hi"),
        ("मुझे Python और Machine Learning सीखना है।", "hi"),
        ("यह email@example.com पर भेजें।", "hi"),
    ]
    
    print("\nFull preprocessing tests:")
    for text, lang in test_cases:
        processed = tokenizer.preprocess_text(text, lang)
        print(f"Input: {text}")
        print(f"Output: {processed}")
        print("-" * 50)


if __name__ == "__main__":
    print("Running Hindi-focused tests...\n")
    test_expand_numbers_multilingual()
    test_abbreviations_multilingual()
    test_symbols_multilingual()
    test_hindi_cleaners()
    test_full_preprocessing()
