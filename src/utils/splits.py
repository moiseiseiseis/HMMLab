# ============================================
# src/utils/splits.py
# ============================================

import json
import random


def create_subject_split(
    subjects,
    test_size=0.2,
    seed=42
):

    random.seed(seed)

    subjects = list(set(subjects))

    random.shuffle(subjects)

    n_test = int(len(subjects) * test_size)

    test_subjects = subjects[:n_test]

    train_subjects = subjects[n_test:]

    return {
        "train": train_subjects,
        "test": test_subjects
    }


def save_split(split, output_path):

    with open(output_path, "w") as f:
        json.dump(split, f, indent=4)