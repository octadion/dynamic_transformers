#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List, Any
from datasets import load_dataset
import pandas as pd
from collections import Counter
import numpy as np
import time

class DatasetInspector:
    """Tools for inspecting and analyzing evaluation datasets."""
    
    def __init__(self):
        self.dataset_configs = {
            "piqa": {"name": "piqa", "has_context": False},
            "arc_easy": {"name": "ai2_arc", "config": "ARC-Easy", "has_context": False},
            "sciq": {"name": "sciq", "has_context": True},
            "winogrande": {"name": "winogrande", "config": "winogrande_xl", "has_context": False},
            "hellaswag": {"name": "hellaswag", "has_context": True}
        }

    def _get_example_details(self, dataset_name: str, ex: dict) -> dict:
        details = {}
        try:
            if dataset_name == "piqa":
                details = {"question": ex["goal"], "choices": [ex["sol1"], ex["sol2"]], "answer_idx": int(ex["label"]), "context": ""}
            elif dataset_name == "arc_easy":
                answer = ex["answerKey"]
                ans_idx = ord(answer) - ord('A') if not answer.isdigit() else int(answer) - 1
                details = {"question": ex["question"], "choices": ex["choices"]["text"], "answer_idx": ans_idx, "context": ""}
            elif dataset_name == "sciq":
                choices = [ex["distractor1"], ex["distractor2"], ex["distractor3"], ex["correct_answer"]]
                correct_text = ex["correct_answer"]
                choices.sort()
                details = {"question": ex["question"], "choices": choices, "answer_idx": choices.index(correct_text), "context": ex.get("support", "")}
            elif dataset_name == "winogrande":
                details = {"question": ex["sentence"], "choices": [ex["option1"], ex["option2"]], "answer_idx": int(ex["answer"]) - 1, "context": ""}
            elif dataset_name == "hellaswag":
                details = {"question": ex["ctx"], "choices": ex["endings"], "answer_idx": int(ex["label"]), "context": ex.get("activity_label", "")}
            
            if "answer_idx" in details:
                details["correct_answer_text"] = details["choices"][details["answer_idx"]]
            return details
        except KeyError as e:
            raise KeyError(f"Schema for '{dataset_name}' doesn't match, missing column: {e}")

    def _load_dataset_with_feedback(self, dataset_name: str, split: str) -> Any:
        config = self.dataset_configs.get(dataset_name)
        if not config:
            raise ValueError(f"Configuration for '{dataset_name}' not found.")
        
        print(f"[INFO] Loading dataset '{dataset_name}' (split: {split})... This might take some time.")
        start_time = time.time()
        
        dataset_args = {"path": config["name"], "split": split}
        if "config" in config:
            dataset_args["name"] = config["config"]
            
        dataset = load_dataset(**dataset_args)
        print(f"[INFO] Dataset '{dataset_name}' loaded successfully in {time.time() - start_time:.2f} seconds.")
        return dataset

    def inspect_dataset(self, dataset_name: str, split: str = "validation", num_samples: int = 5):
        print(f"--- Starting Inspection for '{dataset_name}' ---")
        try:
            dataset = self._load_dataset_with_feedback(dataset_name, split)
            print(f"\nDisplaying first {num_samples} samples:")
            print("-" * 80)
            
            for i, ex in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
                details = self._get_example_details(dataset_name, ex)
                print(f"\nExample {i+1}:")
                if details.get("context"):
                    print(f"Context: {details['context']}")
                print(f"Question: {details['question']}")
                print("Choices:")
                for j, choice in enumerate(details['choices']):
                    marker = "*" if j == details['answer_idx'] else " "
                    print(f"  {marker} {chr(65+j)}. {choice}")
        except Exception as e:
            print(f"[ERROR] Failed to inspect '{dataset_name}': {e}")
        finally:
            print(f"--- Inspection Complete ---")

    def validate_dataset_format(self, dataset_name: str, split: str = "validation") -> List[str]:
        print(f"--- Starting Validation for '{dataset_name}' ---")
        issues = []
        try:
            dataset = self._load_dataset_with_feedback(dataset_name, split)
            print("[INFO] Starting validation for each sample...")
            
            for i, ex in enumerate(dataset):
                try:
                    details = self._get_example_details(dataset_name, ex)
                    if "question" not in details or not details["question"]: issues.append(f"E{i}: Empty question")
                    if "choices" not in details or len(details["choices"]) < 2: issues.append(f"E{i}: Invalid choices")
                    if "answer_idx" not in details or details["answer_idx"] >= len(details["choices"]): issues.append(f"E{i}: Answer index out of range")
                except Exception as e:
                    issues.append(f"E{i}: Failed to process sample: {e}")
                if len(issues) > 10:
                    issues.append("... (too many errors, validation stopped)")
                    break
        except Exception as e:
            issues.append(f"FATAL: Failed to load or process dataset: {e}")
        
        print("-" * 20)
        if not issues:
            print(f"✓ Validation for '{dataset_name}' SUCCESSFUL.")
        else:
            print(f"✗ Validation for '{dataset_name}' FAILED. Issues:")
            for issue in issues:
                print(f"  - {issue}")
        print(f"--- Validation Complete ---")
        return issues

    def compare_datasets(self, dataset_names: list, split: str = "validation"):
        all_stats = {}
        for name in dataset_names:
            try:
                # We don't need to print samples during comparison
                stats = self.inspect_dataset(name, split, num_samples=0)
                all_stats[name] = stats
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
        
        comparison_data = []
        for name, stats in all_stats.items():
            # Manually calculate these stats for the comparison table
            dataset = load_dataset(stats['dataset_name'], name=self.dataset_configs[name].get("config"), split=split)
            questions = [self._get_example_details(name, ex)['question'] for ex in dataset]
            num_choices = len(self._get_example_details(name, dataset[0])['choices'])
            
            comparison_data.append({
                "Dataset": name,
                "Samples": stats["total_samples"],
                "Avg Question Length": f"{np.mean([len(q.split()) for q in questions]):.1f}",
                "Num Choices": num_choices,
                "Has Context": "Yes" if stats["has_context"] else "No"
            })
        
        df = pd.DataFrame(comparison_data)
        print("\nDataset Comparison:")
        print("-" * 80)
        print(df.to_string(index=False))
        return all_stats

    def export_dataset_samples(self, dataset_name: str, output_file: str, split: str = "validation", num_samples: int = 100):
        print(f"Exporting {num_samples} samples from {dataset_name} to {output_file}...")
        config = self.dataset_configs[dataset_name]
        dataset_args = {"path": config["name"], "split": split}
        if "config" in config:
            dataset_args["name"] = config["config"]
        
        dataset = load_dataset(**dataset_args)
        
        samples_to_export = []
        for i, ex in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
            details = self._get_example_details(dataset_name, ex)
            details["index"] = i
            samples_to_export.append(details)
            
        with open(output_file, 'w') as f:
            json.dump(samples_to_export, f, indent=2)
            
        print(f"✓ Export complete.")