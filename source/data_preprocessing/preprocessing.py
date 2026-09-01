import sys
import os
import re
import json
import random
import zipfile
import contractions
import dateparser
from word2number import w2n
from sklearn.model_selection import train_test_split

from source.data_preprocessing.config import (
    DEFAULT_EVAL_PATH,
    DEFAULT_FINE_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_ZIP_PATH,
)

def load_and_preprocess_multiwoz(
    zip_path=DEFAULT_ZIP_PATH,
    sample_size=500, # None
    train_ratio=0.5,
    fine_ratio=0.3,
    eval_ratio=0.2,
    random_seed=42,
    train_output_path=DEFAULT_TRAIN_PATH,
    fine_output_path=DEFAULT_FINE_PATH,
    eval_output_path=DEFAULT_EVAL_PATH
):
    # Extract and load raw data
    destination_dir = "MultiWOZ-coref-extract"
    dataset_dir = os.path.join(destination_dir, "MultiWOZ2_3")
    data_file = os.path.join(dataset_dir, "data.json")
    ontology_file = os.path.join(dataset_dir, "ontology.json")
    dialogue_acts_file = os.path.join(dataset_dir, "dialogue_acts.json")

    os.makedirs(destination_dir, exist_ok=True)
    if not os.path.exists(data_file):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)

    with open(data_file, 'r') as f:
        data = json.load(f)
    with open(ontology_file, 'r') as f:
        ontology = json.load(f)
    with open(dialogue_acts_file, 'r') as f:
        dialogue_acts = json.load(f)

    # Helper functions
    def get_primary_domain(dialogue):
        for key in ("new_goal", "goal"):
            domains = [d for d in dialogue.get(key, {}) if d != "user_action" and isinstance(dialogue[key].get(d), dict)]
            if domains:
                return domains[0]
        domains = set()
        for turn in dialogue.get("log", []):
            for domain in turn.get("metadata", {}):
                if domain != "user_action" and isinstance(turn["metadata"].get(domain), dict):
                    domains.add(domain)
        return domains.pop() if domains else "unknown"

    def normalize_dialogue_id(dialogue_id):
        return dialogue_id[:-len('.json')] if dialogue_id.endswith('.json') else dialogue_id

    # Build raw list
    raw_data = []
    for dialogue_id, dialogue in data.items():
        raw_data.append({
            "dialogue_id": dialogue_id,
            "dialogue": dialogue,
            "domain": get_primary_domain(dialogue),
            "dialogue_acts": dialogue_acts.get(normalize_dialogue_id(dialogue_id), {}),
            "ontology": ontology
        })

    # Sample if requested
    if sample_size is not None and sample_size < len(raw_data):
        raw_data = random.sample(raw_data, sample_size)

    # Split into three parts (stratified by domain)
    # First split: train_fine (train+fine) vs eval
    train_fine_raw, eval_raw = train_test_split(
        raw_data,
        test_size=eval_ratio,
        stratify=[d["domain"] for d in raw_data],
        random_state=random_seed
    )
    # Second split: train vs fine (within train_fine)
    # The proportion of fine within train_fine should be fine_ratio / (train_ratio + fine_ratio)
    fine_proportion_within_train_fine = fine_ratio / (train_ratio + fine_ratio)
    train_raw, fine_raw = train_test_split(
        train_fine_raw,
        test_size=fine_proportion_within_train_fine,
        stratify=[d["domain"] for d in train_fine_raw],
        random_state=random_seed
    )

    print(f"Split sizes: Train={len(train_raw)}, Fine={len(fine_raw)}, Eval={len(eval_raw)}")

    # Normalization functions
    def normalize_general_text(text):
        text = contractions.fix(text, slang=False)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.?!,])', r' \1', text)
        return text

    def normalize_time_in_text(text):
        def time_repl_hhmm_ampm(match):
            parsed = dateparser.parse(match.group(0))
            return parsed.strftime('%H:%M') if parsed else match.group(0)
        text = re.sub(r'\b(\d{1,2}:\d{2})\s*(am|pm)\b', time_repl_hhmm_ampm, text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d{1,2})\s*(am|pm)\b', time_repl_hhmm_ampm, text, flags=re.IGNORECASE)
        text = re.sub(r'\bnoon\b', '12:00', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmidday\b', '12:00', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmidnight\b', '00:00', text, flags=re.IGNORECASE)
        return text

    CURRENCY_SYMBOLS_MAP = {'£': 'GBP', '$': 'USD', '€': 'EUR', '¥': 'JPY', '₹': 'INR'}
    CURRENCY_KEYWORDS_MAP = {
        'pound': 'GBP', 'pounds': 'GBP', 'quid': 'GBP', 'gbp': 'GBP', 'sterling': 'GBP',
        'dollar': 'USD', 'dollars': 'USD', 'buck': 'USD', 'bucks': 'USD', 'usd': 'USD',
        'euro': 'EUR', 'euros': 'EUR', 'eur': 'EUR',
        'yen': 'JPY', 'jpy': 'JPY',
        'rupee': 'INR', 'rupees': 'INR', 'inr': 'INR',
    }
    ALL_CURRENCY_KEYWORDS_SORTED = sorted(CURRENCY_KEYWORDS_MAP.keys(), key=len, reverse=True)
    ALL_CURRENCY_SYMBOLS_SORTED = sorted(CURRENCY_SYMBOLS_MAP.keys(), key=len, reverse=True)
    NUMBER_WORDS_FOR_REGEX = (
        r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion"
    )
    COMPLEX_NUMBER_WORDS_PATTERN = rf"\b((?:{NUMBER_WORDS_FOR_REGEX})(?:\s+(?:and\s+)?(?:{NUMBER_WORDS_FOR_REGEX}))*)\b"

    def normalize_currency_in_text(text):
        def is_valid_number_word_sequence(s):
            words = s.lower().split()
            number_words_set = set(NUMBER_WORDS_FOR_REGEX.split('|'))
            return all(w in number_words_set for w in words)

        def word_num_keyword_replacer(match):
            num_word_str = match.group(1)
            currency_key_str = match.group(2).lower()
            if is_valid_number_word_sequence(num_word_str):
                try:
                    num_val = w2n.word_to_num(num_word_str)
                    currency_code = CURRENCY_KEYWORDS_MAP.get(currency_key_str, currency_key_str.upper())
                    return f"{num_val} {currency_code}"
                except ValueError:
                    pass
            return match.group(0)

        currency_keywords_regex = "|".join([re.escape(k) for k in ALL_CURRENCY_KEYWORDS_SORTED])
        pattern = rf"({COMPLEX_NUMBER_WORDS_PATTERN})\s+({currency_keywords_regex})\b"
        text = re.sub(pattern, word_num_keyword_replacer, text, flags=re.IGNORECASE)
        for symbol in ALL_CURRENCY_SYMBOLS_SORTED:
            code = CURRENCY_SYMBOLS_MAP[symbol]
            text = re.sub(rf'{re.escape(symbol)}\s*(\d+\.?\d*)', rf'\1 {code}', text)
            text = re.sub(rf'(\d+\.?\d*)\s*{re.escape(symbol)}', rf'\1 {code}', text)
        for keyword in ALL_CURRENCY_KEYWORDS_SORTED:
            code = CURRENCY_KEYWORDS_MAP[keyword]
            text = re.sub(rf'(\d+\.?\d*)\s+{re.escape(keyword)}\b', rf'\1 {code}', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(\d)([A-Z]{3}\b)', r'\1 \2', text)
        text = re.sub(r'(\b[A-Z]{3})(\d)', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Process dialogues
    def process_dialogue_turns(dialogue_data):
        turns = dialogue_data.get("log", [])
        processed_turns = []
        for turn_idx, turn in enumerate(turns):
            normalized_text = normalize_currency_in_text(
                normalize_time_in_text(
                    normalize_general_text(turn["text"])
                )
            )
            processed_turns.append({
                "turn_id": turn_idx,
                "speaker": "user" if turn_idx % 2 == 0 else "system",
                "text": normalized_text
            })
        return processed_turns

    def process_data(raw_data_subset):
        return [{
            "dialogue_id": item["dialogue_id"],
            "domain": item["domain"],
            "dialogue": process_dialogue_turns(item["dialogue"])
        } for item in raw_data_subset]

    train_data = process_data(train_raw)
    fine_data = process_data(fine_raw)
    eval_data = process_data(eval_raw)

    # Create User/System pairs from processed dialogues
    def prepare_final_dataset(processed_dialogues):
        sequences = []
        for dialogue_item in processed_dialogues:
            turns = dialogue_item["dialogue"]
            for i in range(0, len(turns) - 1, 2):
                if i + 1 < len(turns):
                    user_turn = turns[i]
                    system_turn = turns[i + 1]
                    if user_turn["speaker"] == "user" and system_turn["speaker"] == "system":
                        formatted_text = f"User: {user_turn['text']}\nSystem: {system_turn['text']}"
                        sequences.append({"text": formatted_text})
        return sequences

    train_sequences = prepare_final_dataset(train_data)
    fine_sequences = prepare_final_dataset(fine_data)
    eval_sequences = prepare_final_dataset(eval_data)

    # Save to files
    def save_sequences_to_file(sequences, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for seq in sequences:
                f.write(seq["text"] + "\n")

    save_sequences_to_file(train_sequences, train_output_path)
    save_sequences_to_file(fine_sequences, fine_output_path)
    save_sequences_to_file(eval_sequences, eval_output_path)

    print(f"Saved: {len(train_sequences)} train, {len(fine_sequences)} fine, {len(eval_sequences)} eval")

    return train_sequences, fine_sequences, eval_sequences


if __name__ == "__main__":
    print("Starting Data Preprocessing....")
    load_and_preprocess_multiwoz()
    print("Data Preprocessing Complete!")