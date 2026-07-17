import os
import tiktoken

FOLDER_PATH = "relevant_texts"
SEP = "=" * 80

total_tokens = 0

all_files = sorted(os.listdir(FOLDER_PATH))
enc = tiktoken.encoding_for_model('gpt-4o')

for i, filename in enumerate(all_files):
    path = os.path.join(FOLDER_PATH, filename)
    if not os.path.isfile(path):
        continue

    print("=" * 80)
    print(f'{i+1} / {len(all_files)}', filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    parts = text.split(SEP, 1)
    if len(parts) == 2:
        text = parts[1].lstrip()
    
    n_tokens = len(enc.encode(text))
    total_tokens += n_tokens
    print(f"\nTokens (this file): {n_tokens}")

print("=" * 80)
print(f"Total tokens (printed files): {total_tokens}")
