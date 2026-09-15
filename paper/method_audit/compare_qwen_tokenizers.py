"""Compare cached, pinned base/embedding token protocols. No downloads or training."""

import json
from pathlib import Path

from transformers import AutoTokenizer


REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}


def canonical_terminal_ids(tokenizer, text, max_length):
    """Proposed raw-text Qwen protocol, isolated from production adapters.

    Reserve a slot, truncate content, then append the designated readout token.
    This is not an adapter for arbitrary chat templates or other model families.
    """
    content = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=max_length-1)["input_ids"]
    return content + [151643]


def main():
    results = {}
    for name, revision in REVISIONS.items():
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=True)
        cases = {
            "default": ("test", {}),
            "manual_pad": ("test"+tokenizer.pad_token, {}),
            "manual_pad_no_special": ("test"+tokenizer.pad_token, {"add_special_tokens": False}),
            "long_manual_pad": ("test "*100+tokenizer.pad_token, {"truncation": True, "max_length": 16}),
            "long_default": ("test "*100, {"truncation": True, "max_length": 16}),
        }
        examples = {}
        for label, (text, options) in cases.items():
            encoded = tokenizer(text, **options)
            examples[label] = dict(ids=encoded["input_ids"][-6:],
                                   attention_mask_tail=encoded["attention_mask"][-6:],
                                   total_tokens=len(encoded["input_ids"]))
        examples["canonical_short"] = canonical_terminal_ids(tokenizer, "test", 16)
        examples["canonical_truncated_tail"] = canonical_terminal_ids(tokenizer, "test "*100, 16)[-6:]
        native_matches = []
        for text in ("", "test", "中文查询：如何优化排序？", "test "*100):
            canonical = canonical_terminal_ids(tokenizer, text, 16)
            native = tokenizer(text, truncation=True, max_length=16)["input_ids"]
            native_matches.append(canonical == native)
            assert len(canonical) <= 16 and canonical[-1] == 151643
        if "Embedding" in name:
            assert all(native_matches), native_matches
        results[name] = dict(revision=revision, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             post_processor=str(tokenizer.backend_tokenizer.post_processor),
                             examples=examples,
                             native_matches_canonical_on_empty_short_unicode_long=native_matches)
    destination = Path(__file__).with_name("tokenizer_comparison.json")
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
