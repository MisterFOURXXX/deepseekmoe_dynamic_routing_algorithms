import os
import re
import json
import random
import zipfile
import contractions
import dateparser
from word2number import w2n
from sklearn.model_selection import train_test_split

def load_and_preprocess_multiwoz(zip_path="MultiWOZ-coref/MultiWOZ2_3.zip", sample_size=300, random_seed=42):
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

    raw_data = []
    for dialogue_id, dialogue in data.items():
        raw_data.append({
            "dialogue_id": dialogue_id,
            "dialogue": dialogue,
            "domain": get_primary_domain(dialogue),
            "dialogue_acts": dialogue_acts.get(normalize_dialogue_id(dialogue_id), {}),
            "ontology": ontology
        })

    raw_data = random.sample(raw_data, sample_size)

    train_val_raw, test_raw = train_test_split(
        raw_data,
        test_size=0.2,
        stratify=[d["domain"] for d in raw_data],
        random_state=random_seed
    )
    train_raw, val_raw = train_test_split(
        train_val_raw,
        test_size=0.1875,
        stratify=[d["domain"] for d in train_val_raw],
        random_state=random_seed
    )

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
        # Validate that the matched string consists only of allowed number words
        def is_valid_number_word_sequence(s):
            # simple check: split and verify each word is in the number words set
            words = s.lower().split()
            number_words_set = set(NUMBER_WORDS_FOR_REGEX.split('|'))
            return all(w in number_words_set for w in words)

        def word_num_keyword_replacer(match):
            num_word_str = match.group(1)
            currency_key_str = match.group(2).lower()
            # Only convert if the number word sequence is valid
            if is_valid_number_word_sequence(num_word_str):
                try:
                    num_val = w2n.word_to_num(num_word_str)
                    currency_code = CURRENCY_KEYWORDS_MAP.get(currency_key_str, currency_key_str.upper())
                    return f"{num_val} {currency_code}"
                except ValueError:
                    pass
            # fallback: leave unchanged
            return match.group(0)

        # rest of the function unchanged except the fallback
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
    val_data = process_data(val_raw)
    test_data = process_data(test_raw)

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
    val_sequences = prepare_final_dataset(val_data)
    test_sequences = prepare_final_dataset(test_data)

    def save_sequences_to_file(sequences, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for seq in sequences:
                f.write(seq["text"] + "\n")

    save_sequences_to_file(train_sequences, "train_sequences.txt")
    save_sequences_to_file(val_sequences, "val_sequences.txt")
    save_sequences_to_file(test_sequences, "test_sequences.txt")

    return train_sequences, val_sequences, test_sequences