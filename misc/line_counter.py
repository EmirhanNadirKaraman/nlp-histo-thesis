import pathlib

def count_lines_in_codebase(directory=".", extensions=(".py", ".sh"), test_dirs=("tests",)):
    total_lines = 0
    file_counts = {ext: 0 for ext in extensions}
    line_counts = {ext: 0 for ext in extensions}
    test_file_counts = {ext: 0 for ext in extensions}
    test_line_counts = {ext: 0 for ext in extensions}

    ignore_dirs = {'.git', 'venv', '.venv', 'env', '__pycache__', 'node_modules'}

    base_path = pathlib.Path(directory)

    for file_path in base_path.rglob('*'):
        if any(ignored in file_path.parts for ignored in ignore_dirs):
            continue

        if file_path.suffix in extensions:
            try:
                with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)

                is_test = any(part in test_dirs for part in file_path.parts) or file_path.name.startswith("test_")
                line_counts[file_path.suffix] += lines
                file_counts[file_path.suffix] += 1
                total_lines += lines
                if is_test:
                    test_line_counts[file_path.suffix] += lines
                    test_file_counts[file_path.suffix] += 1
            except Exception:
                continue

    print(f"{'Extension':<12} | {'Files':<8} | {'Lines':<10} | {'TestFiles':<10} | {'TestLines':<10}")
    print("-" * 65)
    for ext in extensions:
        print(f"{ext:<12} | {file_counts[ext]:<8} | {line_counts[ext]:<10} | {test_file_counts[ext]:<10} | {test_line_counts[ext]:<10}")
    print("-" * 65)
    print(
        f"{'TOTAL':<12} | {sum(file_counts.values()):<8} | {total_lines:<10} | "
        f"{sum(test_file_counts.values()):<10} | {sum(test_line_counts.values()):<10}"
    )
    non_test_total = total_lines - sum(test_line_counts.values())
    print(f"{'NON-TEST':<12} | {sum(file_counts.values()) - sum(test_file_counts.values()):<8} | {non_test_total:<10}")

if __name__ == "__main__":
    count_lines_in_codebase()
