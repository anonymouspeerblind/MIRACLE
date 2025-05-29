import json
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import re
import pycld2 as cld2
from googletrans import Translator

translator = Translator()
DetectorFactory.seed = 0

def is_english(text):
    try:
        # detect returns a language code, e.g., 'en' for English
        lang = detect(text)
        return lang == 'en'
    except Exception as e:
        print(f"Error detecting language: {e}")
        return False

def has_mixed_language(text):
    try:
        # Detect languages in the text
        is_reliable, text_bytes_found, details = cld2.detect(text)
        # details is a list of tuples: (language_name, language_code, percent, score)
        languages = {lang[1] for lang in details}  # Use a set to get unique language codes
        
        # Check if more than one language is detected and if any language is not English.
        if len(languages) > 1 and any(lang != 'EN' for lang in languages):
            return True, languages
        else:
            return False, languages
    except Exception as e:
        print("Error during detection:", e)
        return False, set()

def has_mandarin(text):
    # This regex pattern matches any Chinese character
    pattern = r'[\u4e00-\u9fff]'
    return bool(re.search(pattern, text))

def translate_mandarin_parts(text):
    # This regex matches one or more consecutive Mandarin characters.
    pattern = r'([\u4e00-\u9fff]+)'
    
    # Find all segments in the text that contain Mandarin.
    mandarin_segments = re.findall(pattern, text)
    
    # For each Mandarin segment, translate it and replace it in the original text.
    for segment in mandarin_segments:
        # Translate the segment (assuming the source is Chinese and destination is English)
        translated = translator.translate(segment, src='zh-cn', dest='en').text
        # Replace the segment with its English translation.
        text = text.replace(segment, " "+translated)
    return text

if __name__ == "__main__":
    with open("<path to LLM remarks>", "r") as js:
        rem_ds = json.load(js)
    print(f"length of DS samples: {len(rem_ds.keys())}")
    for i in tqdm(rem_ds):
        if len(rem_ds[i]['remarks']) < 100:
            print("low length")
            print(rem_ds[i])
        if not rem_ds[i]['remarks'].isascii():
            print(i)