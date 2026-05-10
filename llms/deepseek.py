import os
import json
import requests
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

API_KEY = ""  # Your API Key
ENDPOINT = ""
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

DATASET_PROMPTS = {
    'ImageNet': {
        'parent_class': "Based on function or form, assign {class_name} to a broad object category (≤5 words):",
    },
    'OxfordPets': {
        'parent_class': "According to biological traits, assign {class_name} to an animal family (≤5 words):",
    },
    'Caltech101': {
        'parent_class': "By functional use, assign {class_name} to an object category (≤5 words):",
    },
    'DescribableTextures': {
        'parent_class': "Based on its origin or visual characteristics, assign {class_name} to a texture category (≤5 words):",
    },
    'EuroSAT': {
        'parent_class': "By land use and dominant observable surface signature, assign {class_name} to a land category (≤5 words):",
    },
    'FGVCAircraft': {
        'parent_class': "By aviation role and main mission, assign {class_name} to an aircraft category (≤5 words):",
    },
    'Food101': {
        'parent_class': "In everyday dining, how would people casually group {class_name} on a menu? (≤5 words):",
    },
    'OxfordFlowers': {
        'parent_class': "By botanical classification, assign {class_name} to a plant family (≤5 words):",
    },
    'StanfordCars': {
        'parent_class': "By design style, body shape or function (e.g. sedan/SUV), assign {class_name} to an automotive category (≤5 words):"
    },
    'SUN397': {
        'parent_class': "By environment type, assign {class_name} to a scene category (≤5 words):",
    },
    'UCF101': {
        'parent_class': "According to who performs the action and what they interact with, assign {class_name} to a action cluster (≤5 words): ",
    }
}

SYSTEM_PROMPT = (
    "You are an expert classifier. Answer concisely with only the requested information. "
    "Do not include any explanations, introductory phrases or additional text. "
    "Strictly follow the word count limits and specific instructions in each prompt."
)


def generate_category(dataset_name: str, classname: str,
                      max_retries: int = 3, timeout: int = 30) -> dict:
    prompts = DATASET_PROMPTS[dataset_name]
    category_prompt = prompts['parent_class'].format(class_name=classname)

    parent_class = ask_deepseek(category_prompt, max_tokens=20,
                                max_retries=max_retries, timeout=timeout)

    return {'parent_class': validate_output(parent_class, max_words=5)}


def ask_deepseek(prompt: str,
                 max_tokens: int = 50,
                 max_retries: int = 3,
                 timeout: int = 30) -> str:
    dialog = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "deepseek-chat",
        "messages": dialog,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": max_tokens
    }

    for attempt in range(1, max_retries + 1):
        try:
            rsp = requests.post(ENDPOINT, headers=headers,
                                json=payload, timeout=timeout)
            rsp.raise_for_status()
            content = rsp.json()['choices'][0]['message']['content'].strip()
            if ':' in content:
                content = content.split(':', 1)[-1].strip()
            return content
        except Exception as e:
            logging.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"DeepSeek API finally failed for prompt: {prompt}"
                ) from e
            time.sleep(1)


def validate_output(text: str, max_words: int) -> str:
    text = text.strip().strip('"').strip()
    if not text:
        raise ValueError("Empty response from API")
    forbidden_prefixes = ["category", "family", "type", "class"]
    for prefix in forbidden_prefixes:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip(" :.-")
    words = text.split()
    if len(words) > max_words:
        logging.warning("Output truncated to %d words", max_words)
    return text


def process_dataset(dataset_name: str, class_file_path: str, output_dir: str):
    if not os.path.exists(class_file_path):
        logging.error("Class file not found: %s", class_file_path)
        return
    with open(class_file_path, 'r', encoding='utf-8') as f:
        classnames = [name.strip()
                      for name in f.read().splitlines() if name.strip()]
    result = {}
    for classname in tqdm(classnames, desc=f"Processing {dataset_name}"):
        result[classname] = generate_category(dataset_name, classname)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    logging.info("Descriptions saved to: %s", output_file)
    if classnames:
        example_class = classnames[0]
        pc = result[example_class]['parent_class']
        if pc:
            logging.info("Example for %s - %s:", dataset_name, example_class)
            logging.info("  parent_class: %s", pc)


def main():
    output_dir = '../prompt'
    for dataset_name in DATASET_PROMPTS.keys():
        class_file_path = f'classname/{dataset_name}.txt'
        logging.info(f"Processing dataset: {dataset_name}")
        process_dataset(dataset_name, class_file_path, output_dir)


if __name__ == "__main__":
    main()
