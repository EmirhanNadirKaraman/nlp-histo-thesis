"""
Count tokens in sentences from disease entity JSON files.

Calculates token counts for each sentence in each JSON file in a directory.
Uses tiktoken (OpenAI tokenizer) by default.
Includes cost estimation for LLM API calls.

Usage:
    python legacy/langchain-summarization/count_tokens.py
    python legacy/langchain-summarization/count_tokens.py --input-dir disease_entities
    python legacy/langchain-summarization/count_tokens.py --estimate-cost
    python legacy/langchain-summarization/count_tokens.py --output stats.json
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def count_tokens_whitespace(text: str) -> int:
    """Simple whitespace tokenizer."""
    return len(text.split())


def count_tokens_tiktoken(text: str, encoding) -> int:
    """OpenAI tiktoken tokenizer (for GPT models)."""
    return len(encoding.encode(text))


# Pricing per 1M tokens (input, output) - Updated Jan 2025
# Batch API: 50% discount on all tokens (async, 24hr turnaround)
MODEL_PRICING = {
    "gpt-4o": {
        "input": 2.50, "output": 10.00, "cached_input": 1.25,
        "batch_input": 1.25, "batch_output": 5.00
    },
    "gpt-4o-mini": {
        "input": 0.15, "output": 0.60, "cached_input": 0.075,
        "batch_input": 0.075, "batch_output": 0.30
    },
}

# Prompt token estimates (from langchain_summarization.ipynb)
# Counted via tiktoken on actual prompts
PROMPT_TOKENS = {
    "map_system": 450,      # MAP_SYSTEM_PROMPT (cacheable)
    "map_user": 50,         # MAP_USER_PROMPT template (variable)
    "reduce_system": 550,   # REDUCE_SYSTEM_PROMPT (cacheable)
    "reduce_user": 50,      # REDUCE_USER_PROMPT template (variable)
    "rule_system": 400,     # RULE_EXTRACTION_SYSTEM_PROMPT (cacheable)
    "rule_user": 50,        # RULE_EXTRACTION_USER_PROMPT template (variable)
}

# Output token estimates (structured JSON Pydantic models)
OUTPUT_TOKENS = {
    "map": 400,             # AuditableSummary per chunk
    "reduce": 800,          # ConsolidatedSummary
    "rules": 600,           # ExtractedRules
}


def estimate_pipeline_cost(
    total_input_tokens: int,
    num_sentences: int,
    num_files: int,
    chunk_size: int = 10
) -> dict:
    """
    Estimate cost for the langchain map-reduce-rules pipeline.

    Pipeline (runs independently for EACH FILE):
    1. MAP (gpt-4o-mini): Chunks of 10 sentences -> AuditableSummary
    2. REDUCE (gpt-4o): Combine summaries -> ConsolidatedSummary (recursive)
    3. RULES (gpt-4o): Extract rules from final summary

    Args:
        total_input_tokens: Total tokens across all sentences
        num_sentences: Total number of sentences
        num_files: Number of concept files (each file = one full pipeline run)
        chunk_size: Sentences per chunk (default: 10)

    Returns:
        dict with detailed cost breakdown
    """
    mini_price = MODEL_PRICING["gpt-4o-mini"]
    gpt4o_price = MODEL_PRICING["gpt-4o"]

    # Calculate totals and per-file averages
    num_chunks = (num_sentences + chunk_size - 1) // chunk_size
    avg_tokens_per_sentence = total_input_tokens / num_sentences if num_sentences > 0 else 0
    avg_tokens_per_chunk = avg_tokens_per_sentence * chunk_size
    avg_sentences_per_file = num_sentences / num_files if num_files > 0 else 0
    avg_chunks_per_file = (avg_sentences_per_file + chunk_size - 1) // chunk_size if avg_sentences_per_file > 0 else 1

    # === MAP PHASE (gpt-4o-mini) ===
    # Each chunk: system prompt (cacheable) + user prompt + chunk tokens
    # First call: full price, subsequent calls: system prompt is cached at 50% discount
    map_system_tokens = PROMPT_TOKENS["map_system"]
    map_variable_tokens = PROMPT_TOKENS["map_user"] + avg_tokens_per_chunk

    # First call pays full price, rest get cached system prompt
    map_first_call_input = map_system_tokens + map_variable_tokens
    map_cached_calls_input = map_variable_tokens * (num_chunks - 1) if num_chunks > 1 else 0
    map_cached_system_tokens = map_system_tokens * (num_chunks - 1) if num_chunks > 1 else 0

    map_input_total = map_first_call_input + map_cached_calls_input + map_cached_system_tokens
    map_output_total = OUTPUT_TOKENS["map"] * num_chunks

    # Cost with caching
    map_input_cost = (map_first_call_input / 1_000_000) * mini_price["input"]
    map_input_cost += (map_cached_calls_input / 1_000_000) * mini_price["input"]
    map_input_cost += (map_cached_system_tokens / 1_000_000) * mini_price["cached_input"]
    map_output_cost = (map_output_total / 1_000_000) * mini_price["output"]

    # === REDUCE PHASE (gpt-4o) ===
    # Each FILE runs its own reduce tree independently
    # System prompts cached after first call (within batch, across files)

    def calc_reduce_for_chunks(n_chunks):
        """Calculate reduce calls and tokens for a file with n_chunks."""
        if n_chunks <= 0:
            return 0, 0, 0, 0, 0

        levels = 0
        calls = 0
        variable_tokens = 0  # User prompt + summaries (not cacheable)
        output_tokens = 0
        current = n_chunks
        is_first = True

        while current > 1:
            levels += 1
            batch_calls = (current + 9) // 10
            calls += batch_calls

            if is_first:
                tokens_per = OUTPUT_TOKENS["map"] * 0.7
                is_first = False
            else:
                tokens_per = OUTPUT_TOKENS["reduce"] * 0.7

            variable_tokens += PROMPT_TOKENS["reduce_user"] * batch_calls
            variable_tokens += current * tokens_per
            output_tokens += OUTPUT_TOKENS["reduce"] * batch_calls

            current = batch_calls

        # Even 1 chunk needs 1 reduce call
        if n_chunks >= 1 and calls == 0:
            levels = 1
            calls = 1
            variable_tokens = PROMPT_TOKENS["reduce_user"] + OUTPUT_TOKENS["map"] * 0.7
            output_tokens = OUTPUT_TOKENS["reduce"]

        return levels, calls, variable_tokens, output_tokens

    # Calculate for average file
    avg_reduce_levels, avg_reduce_calls, avg_reduce_variable, avg_reduce_output = calc_reduce_for_chunks(int(avg_chunks_per_file))

    # Total across all files
    reduce_levels = avg_reduce_levels
    reduce_calls_total = avg_reduce_calls * num_files
    reduce_variable_total = avg_reduce_variable * num_files
    reduce_output_total = avg_reduce_output * num_files

    # System prompt: first call full price, rest cached
    reduce_system_tokens = PROMPT_TOKENS["reduce_system"]
    reduce_first_system = reduce_system_tokens  # First call
    reduce_cached_system = reduce_system_tokens * (reduce_calls_total - 1) if reduce_calls_total > 1 else 0

    reduce_input_total = reduce_first_system + reduce_cached_system + reduce_variable_total

    reduce_input_cost = (reduce_first_system / 1_000_000) * gpt4o_price["input"]
    reduce_input_cost += (reduce_cached_system / 1_000_000) * gpt4o_price["cached_input"]
    reduce_input_cost += (reduce_variable_total / 1_000_000) * gpt4o_price["input"]
    reduce_output_cost = (reduce_output_total / 1_000_000) * gpt4o_price["output"]

    # === RULES PHASE (gpt-4o) ===
    # One call per file with final summary
    # System prompt cached after first file
    rules_system_tokens = PROMPT_TOKENS["rule_system"]
    rules_variable_per_file = PROMPT_TOKENS["rule_user"] + OUTPUT_TOKENS["reduce"]

    rules_first_input = rules_system_tokens + rules_variable_per_file
    rules_cached_system = rules_system_tokens * (num_files - 1) if num_files > 1 else 0
    rules_variable_rest = rules_variable_per_file * (num_files - 1) if num_files > 1 else 0

    rules_input_total = rules_first_input + rules_cached_system + rules_variable_rest
    rules_output_total = OUTPUT_TOKENS["rules"] * num_files

    rules_input_cost = (rules_first_input / 1_000_000) * gpt4o_price["input"]
    rules_input_cost += (rules_cached_system / 1_000_000) * gpt4o_price["cached_input"]
    rules_input_cost += (rules_variable_rest / 1_000_000) * gpt4o_price["input"]
    rules_output_cost = (rules_output_total / 1_000_000) * gpt4o_price["output"]

    # === TOTALS ===
    total_cost = (map_input_cost + map_output_cost +
                  reduce_input_cost + reduce_output_cost +
                  rules_input_cost + rules_output_cost)

    # Calculate cost WITHOUT caching for comparison
    total_cost_no_cache = (
        (map_input_total / 1_000_000) * mini_price["input"] +
        (map_output_total / 1_000_000) * mini_price["output"] +
        (reduce_input_total / 1_000_000) * gpt4o_price["input"] +
        (reduce_output_total / 1_000_000) * gpt4o_price["output"] +
        (rules_input_total / 1_000_000) * gpt4o_price["input"] +
        (rules_output_total / 1_000_000) * gpt4o_price["output"]
    )

    cache_savings = total_cost_no_cache - total_cost

    # === BATCH API COST (50% discount, no caching) ===
    batch_map_cost = (
        (map_input_total / 1_000_000) * mini_price["batch_input"] +
        (map_output_total / 1_000_000) * mini_price["batch_output"]
    )
    batch_reduce_cost = (
        (reduce_input_total / 1_000_000) * gpt4o_price["batch_input"] +
        (reduce_output_total / 1_000_000) * gpt4o_price["batch_output"]
    )
    batch_rules_cost = (
        (rules_input_total / 1_000_000) * gpt4o_price["batch_input"] +
        (rules_output_total / 1_000_000) * gpt4o_price["batch_output"]
    )
    total_batch_cost = batch_map_cost + batch_reduce_cost + batch_rules_cost

    return {
        "summary": {
            "total_sentences": num_sentences,
            "total_files": num_files,
            "total_chunks": num_chunks,
            "reduce_levels": reduce_levels,
            "reduce_calls": reduce_calls_total,
            # Real-time API with prompt caching
            "total_cost": round(total_cost, 4),
            "cost_without_cache": round(total_cost_no_cache, 4),
            "cache_savings": round(cache_savings, 4),
            # Batch API (50% discount, async)
            "batch_cost": round(total_batch_cost, 4),
            "batch_map_cost": round(batch_map_cost, 4),
            "batch_reduce_cost": round(batch_reduce_cost, 4),
            "batch_rules_cost": round(batch_rules_cost, 4),
        },
        "map_phase": {
            "model": "gpt-4o-mini",
            "calls": num_chunks,
            "input_tokens": int(map_input_total),
            "output_tokens": int(map_output_total),
            "cost": round(map_input_cost + map_output_cost, 4)
        },
        "reduce_phase": {
            "model": "gpt-4o",
            "levels": reduce_levels,
            "input_tokens": int(reduce_input_total),
            "output_tokens": int(reduce_output_total),
            "cost": round(reduce_input_cost + reduce_output_cost, 4)
        },
        "rules_phase": {
            "model": "gpt-4o",
            "calls": num_files,
            "input_tokens": int(rules_input_total),
            "output_tokens": int(rules_output_total),
            "cost": round(rules_input_cost + rules_output_cost, 4)
        }
    }


def analyze_tokens(
    input_dir: str = "disease_entities",
    output_file: str = None,
    tokenizer: str = "tiktoken",
    estimate_costs: bool = False
):
    """
    Count tokens in all sentences across JSON files.

    Args:
        input_dir: Directory containing JSON files
        output_file: Optional output file for detailed stats
        tokenizer: Tokenizer to use ('tiktoken' or 'whitespace')
        estimate_costs: Whether to show cost estimates for LLM APIs

    Returns:
        dict with statistics
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' not found.")
        return None

    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return None

    print("=" * 80)
    print(f"Token Counter - {tokenizer} tokenizer")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"JSON files found: {len(json_files)}")

    # Load tiktoken if needed
    encoding = None
    if tokenizer == "tiktoken":
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            print("Using tiktoken (cl100k_base encoding)")
        except ImportError:
            print("tiktoken not installed. Run: pip install tiktoken")
            print("Falling back to whitespace tokenizer.")
            tokenizer = "whitespace"

    print("=" * 80 + "\n")

    # Collect stats
    all_token_counts = []
    file_stats = []
    total_sentences = 0

    for json_file in tqdm(sorted(json_files), desc="Processing files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            tqdm.write(f"Error reading {json_file.name}: {e}")
            continue

        sentences = data.get("sentences", [])
        if not sentences:
            continue

        file_token_counts = []
        for sent_data in sentences:
            sentence = sent_data.get("sentence", "")
            if not sentence:
                continue

            if tokenizer == "tiktoken":
                token_count = count_tokens_tiktoken(sentence, encoding)
            else:
                token_count = count_tokens_whitespace(sentence)

            file_token_counts.append(token_count)
            all_token_counts.append(token_count)

        if file_token_counts:
            file_stats.append({
                "file": json_file.name,
                "cui": data.get("umls_cui"),
                "canonical_name": data.get("canonical_name"),
                "sentence_count": len(file_token_counts),
                "total_tokens": sum(file_token_counts),
                "min_tokens": min(file_token_counts),
                "max_tokens": max(file_token_counts),
                "avg_tokens": sum(file_token_counts) / len(file_token_counts),
                "token_counts": file_token_counts
            })
            total_sentences += len(file_token_counts)

    if not all_token_counts:
        print("No sentences found in any files.")
        return None

    # Overall stats
    overall_stats = {
        "total_files": len(file_stats),
        "total_sentences": total_sentences,
        "total_tokens": sum(all_token_counts),
        "min_tokens": min(all_token_counts),
        "max_tokens": max(all_token_counts),
        "avg_tokens": sum(all_token_counts) / len(all_token_counts),
        "median_tokens": sorted(all_token_counts)[len(all_token_counts) // 2],
        "tokenizer": tokenizer
    }

    # Print summary
    print("Overall Statistics")
    print("-" * 80)
    print(f"Total files processed: {overall_stats['total_files']}")
    print(f"Total sentences: {overall_stats['total_sentences']}")
    print(f"Total tokens: {overall_stats['total_tokens']:,}")
    print(f"Min tokens per sentence: {overall_stats['min_tokens']}")
    print(f"Max tokens per sentence: {overall_stats['max_tokens']}")
    print(f"Avg tokens per sentence: {overall_stats['avg_tokens']:.1f}")
    print(f"Median tokens per sentence: {overall_stats['median_tokens']}")

    # Token distribution
    print("\nToken Distribution:")
    ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, float('inf'))]
    for low, high in ranges:
        count = sum(1 for t in all_token_counts if low <= t < high)
        pct = count / len(all_token_counts) * 100
        label = f"{low}-{high}" if high != float('inf') else f"{low}+"
        bar = "█" * int(pct / 2)
        print(f"  {label:>10}: {count:>6} ({pct:>5.1f}%) {bar}")

    # Top files by token count
    print("\nTop 10 Files by Total Tokens:")
    print("-" * 80)
    for stat in sorted(file_stats, key=lambda x: x['total_tokens'], reverse=True)[:10]:
        name = stat['canonical_name'] or stat['cui']
        print(f"  {name[:40]:<40} {stat['sentence_count']:>5} sentences, {stat['total_tokens']:>8} tokens")

    # Cost estimation
    if estimate_costs:
        print("\n" + "=" * 80)
        print("COST ESTIMATION (LangChain Map-Reduce-Rules Pipeline)")
        print("=" * 80)

        costs = estimate_pipeline_cost(
            total_input_tokens=overall_stats['total_tokens'],
            num_sentences=total_sentences,
            num_files=overall_stats['total_files']
        )

        print("\nPipeline Overview:")
        print(f"  Total sentences:  {costs['summary']['total_sentences']:,}")
        print(f"  Total files:      {costs['summary']['total_files']}")
        print(f"  Chunks (10 sent): {costs['summary']['total_chunks']}")
        print(f"  Reduce levels:    {costs['summary']['reduce_levels']} (recursive depth)")
        print(f"  Reduce calls:     {costs['summary']['reduce_calls']} (total API calls)")

        print(f"\n{'Phase':<20} {'Model':<15} {'Calls':<8} {'Input':<12} {'Output':<12}")
        print("-" * 70)

        print(f"{'MAP':<20} {costs['map_phase']['model']:<15} {costs['map_phase']['calls']:<8} "
              f"{costs['map_phase']['input_tokens']:>10,}  {costs['map_phase']['output_tokens']:>10,}")

        print(f"{'REDUCE':<20} {costs['reduce_phase']['model']:<15} {costs['summary']['reduce_calls']:<8} "
              f"{costs['reduce_phase']['input_tokens']:>10,}  {costs['reduce_phase']['output_tokens']:>10,}")

        print(f"{'RULES':<20} {costs['rules_phase']['model']:<15} {costs['rules_phase']['calls']:<8} "
              f"{costs['rules_phase']['input_tokens']:>10,}  {costs['rules_phase']['output_tokens']:>10,}")

        # === OPTION 1: Real-time API with caching ===
        print("\n" + "=" * 80)
        print("OPTION 1: Real-time API (instant results, prompt caching)")
        print("=" * 80)
        print(f"  {'Phase':<15} {'Cost':<12}")
        print(f"  {'-'*30}")
        print(f"  {'MAP':<15} ${costs['map_phase']['cost']:.4f}")
        print(f"  {'REDUCE':<15} ${costs['reduce_phase']['cost']:.4f}")
        print(f"  {'RULES':<15} ${costs['rules_phase']['cost']:.4f}")
        print(f"  {'-'*30}")
        print(f"  {'TOTAL':<15} ${costs['summary']['total_cost']:.4f}")
        print(f"\n  Without caching: ${costs['summary']['cost_without_cache']:.4f}")
        print(f"  Cache savings:   ${costs['summary']['cache_savings']:.4f} ({costs['summary']['cache_savings']/costs['summary']['cost_without_cache']*100:.1f}%)")

        # === OPTION 2: Batch API ===
        print("\n" + "=" * 80)
        print("OPTION 2: Batch API (async, 24hr turnaround, 50% discount)")
        print("=" * 80)
        print(f"  {'Phase':<15} {'Cost':<12}")
        print(f"  {'-'*30}")
        print(f"  {'MAP':<15} ${costs['summary']['batch_map_cost']:.4f}")
        print(f"  {'REDUCE':<15} ${costs['summary']['batch_reduce_cost']:.4f}")
        print(f"  {'RULES':<15} ${costs['summary']['batch_rules_cost']:.4f}")
        print(f"  {'-'*30}")
        print(f"  {'TOTAL':<15} ${costs['summary']['batch_cost']:.4f}")

        # === Comparison ===
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        realtime = costs['summary']['total_cost']
        batch = costs['summary']['batch_cost']
        savings = realtime - batch
        print(f"  Real-time (with caching):  ${realtime:.4f}")
        print(f"  Batch API:                 ${batch:.4f}")
        print(f"  Batch savings:             ${savings:.4f} ({savings/realtime*100:.1f}%)")
        print(f"\n  Best option: {'Batch API' if batch < realtime else 'Real-time'}")

    # Save detailed output if requested
    if output_file:
        output_data = {
            "overall": overall_stats,
            "files": [{k: v for k, v in f.items() if k != 'token_counts'} for f in file_stats]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Detailed stats saved to {output_file}")

    print("\n" + "=" * 80)

    return {
        "overall": overall_stats,
        "files": file_stats
    }


def estimate_cost_with_deduplication(
    input_dir: str = "disease_entities",
    tokenizer: str = "tiktoken",
    verbose: bool = True
) -> dict:
    """
    Estimate cost based on UNIQUE text_element_ids across all input files.

    Since the same paragraph (text_element_id) can appear in multiple concept files,
    this function deduplicates by text_element_id to show the true cost when
    paragraph-level caching is implemented.

    The MAP phase processes sentences in chunks. With caching, each unique
    text_element only needs to be sent to the LLM once, regardless of how
    many concept files it appears in.

    Args:
        input_dir: Directory containing input JSON files
        tokenizer: Tokenizer to use ('tiktoken' or 'whitespace')
        verbose: Print detailed output

    Returns:
        dict with:
            - total_sentences: count across all files (with duplicates)
            - unique_sentences: count of unique text_element_ids
            - duplicate_savings: sentences saved by deduplication
            - total_tokens: tokens for unique sentences only
            - estimated_cost: cost estimate based on unique text elements
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' not found.")
        return None

    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return None

    # Load tiktoken if needed
    encoding = None
    if tokenizer == "tiktoken":
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
        except ImportError:
            print("tiktoken not installed, falling back to whitespace tokenizer")
            tokenizer = "whitespace"

    # Collect unique text elements by text_element_id
    # Key: text_element_id, Value: {sentence, pmcid, tokens}
    unique_text_elements = {}
    total_sentences_with_duplicates = 0
    num_files = len(json_files)

    for json_file in tqdm(sorted(json_files), desc="Scanning files", disable=not verbose):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            sentences = data.get("sentences", [])

            for sent_data in sentences:
                total_sentences_with_duplicates += 1

                text_element_id = sent_data.get("text_element_id")
                sentence = sent_data.get("sentence", "")

                if not sentence or text_element_id is None:
                    continue

                # Only count tokens for first occurrence of each text_element_id
                if text_element_id not in unique_text_elements:
                    if tokenizer == "tiktoken":
                        tokens = count_tokens_tiktoken(sentence, encoding)
                    else:
                        tokens = count_tokens_whitespace(sentence)

                    unique_text_elements[text_element_id] = {
                        "sentence": sentence,
                        "pmcid": sent_data.get("pmcid"),
                        "tokens": tokens
                    }

        except Exception as e:
            if verbose:
                tqdm.write(f"Warning: Could not read {json_file.name}: {e}")

    # Calculate totals
    unique_count = len(unique_text_elements)
    total_tokens = sum(te["tokens"] for te in unique_text_elements.values())
    duplicate_count = total_sentences_with_duplicates - unique_count

    # Estimate cost based on unique text elements
    cost_estimate = None
    if unique_count > 0:
        cost_estimate = estimate_pipeline_cost(
            total_input_tokens=total_tokens,
            num_sentences=unique_count,
            num_files=num_files
        )

    # Also calculate what cost would be WITHOUT deduplication for comparison
    cost_without_dedup = None
    if total_sentences_with_duplicates > 0:
        # Estimate tokens for duplicates (use average tokens per unique sentence)
        avg_tokens = total_tokens / unique_count if unique_count > 0 else 0
        total_tokens_with_duplicates = int(avg_tokens * total_sentences_with_duplicates)

        cost_without_dedup = estimate_pipeline_cost(
            total_input_tokens=total_tokens_with_duplicates,
            num_sentences=total_sentences_with_duplicates,
            num_files=num_files
        )

    result = {
        "total_files": num_files,
        "total_sentences": total_sentences_with_duplicates,
        "unique_sentences": unique_count,
        "duplicate_sentences": duplicate_count,
        "deduplication_ratio": duplicate_count / total_sentences_with_duplicates if total_sentences_with_duplicates > 0 else 0,
        "unique_tokens": total_tokens,
        "estimated_cost": cost_estimate,
        "cost_without_deduplication": cost_without_dedup
    }

    if verbose:
        print("\n" + "=" * 80)
        print("COST ESTIMATE WITH TEXT ELEMENT DEDUPLICATION")
        print("=" * 80)
        print(f"Total files:              {num_files}")
        print(f"Total sentences (raw):    {total_sentences_with_duplicates:,}")
        print(f"Unique text_element_ids:  {unique_count:,}")
        print(f"Duplicate sentences:      {duplicate_count:,} ({result['deduplication_ratio']*100:.1f}% savings)")
        print(f"\nTokens (unique only):     {total_tokens:,}")

        if cost_estimate and cost_without_dedup:
            cost_with = cost_estimate['summary']['total_cost']
            cost_without = cost_without_dedup['summary']['total_cost']
            savings = cost_without - cost_with

            print("\n" + "-" * 80)
            print("COST COMPARISON")
            print("-" * 80)
            print(f"  Without caching (all sentences):  ${cost_without:.4f}")
            print(f"  With caching (unique only):       ${cost_with:.4f}")
            print(f"  Savings from caching:             ${savings:.4f} ({savings/cost_without*100:.1f}%)")

            print("\n" + "-" * 80)
            print("WITH CACHING - DETAILED BREAKDOWN")
            print("-" * 80)
            print(f"  MAP phase (gpt-4o-mini):   ${cost_estimate['map_phase']['cost']:.4f}")
            print(f"  REDUCE phase (gpt-4o):     ${cost_estimate['reduce_phase']['cost']:.4f}")
            print(f"  RULES phase (gpt-4o):      ${cost_estimate['rules_phase']['cost']:.4f}")
            print(f"  TOTAL (real-time API):     ${cost_with:.4f}")
            print(f"  TOTAL (batch API, 50%):    ${cost_estimate['summary']['batch_cost']:.4f}")

        print("=" * 80)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in disease entity sentences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (tiktoken tokenizer)
  python legacy/langchain-summarization/count_tokens.py

  # With cost estimation for your LangChain pipeline
  python legacy/langchain-summarization/count_tokens.py --estimate-cost

  # Estimate cost with paragraph-level deduplication (for caching)
  python legacy/langchain-summarization/count_tokens.py --deduplicate --input-dir relevant_texts

  # Use whitespace tokenizer (faster)
  python legacy/langchain-summarization/count_tokens.py --tokenizer whitespace

  # Custom input directory
  python legacy/langchain-summarization/count_tokens.py --input-dir umls_entities

  # Save detailed stats to file
  python legacy/langchain-summarization/count_tokens.py --output token_stats.json
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='disease_entities',
        help='Directory containing JSON files (default: disease_entities)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        choices=['tiktoken', 'whitespace'],
        default='tiktoken',
        help='Tokenizer to use (default: tiktoken)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for detailed statistics (JSON)'
    )
    parser.add_argument(
        '--estimate-cost',
        action='store_true',
        help='Show cost estimates for LLM API calls (map-reduce-rules pipeline)'
    )
    parser.add_argument(
        '--deduplicate',
        action='store_true',
        help='Estimate cost based on unique text_element_ids (for paragraph-level caching)'
    )

    args = parser.parse_args()

    # If --deduplicate is provided, estimate cost with text element deduplication
    if args.deduplicate:
        estimate_cost_with_deduplication(
            input_dir=args.input_dir,
            tokenizer=args.tokenizer
        )
    else:
        analyze_tokens(
            input_dir=args.input_dir,
            output_file=args.output,
            tokenizer=args.tokenizer,
            estimate_costs=args.estimate_cost
        )


if __name__ == "__main__":
    main()
