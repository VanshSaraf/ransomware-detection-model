import pandas as pd
import os

RANSOMWARE_KEYWORDS = [
    'cerber','gandcrab','sodinokibi','darkside','lockbit',
    'conti','ryuk','wannacry','revil','maze','phobos','cryptolocker'
]

def is_ransomware(class_name):
    return any(k in str(class_name).lower() for k in RANSOMWARE_KEYWORDS)

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    df['is_ransomware'] = df['class'].apply(is_ransomware)
    df['target_label'] = df['is_ransomware'].map({
        True: 'malicious',
        False: 'benign'
    })

    feature_cols = [
        'x0_entropy_write',
        'x1_write_Bps',
        'x2_read_Bps',
        'x3_var_lba_write',
        'x4_var_lba_read'
    ]

    return df, feature_cols
