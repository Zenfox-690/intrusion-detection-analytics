"""Data loading and attack label mapping for NSL-KDD dataset.

Provides utilities to read KDDTrain+.txt and KDDTest+.txt with standard 41 feature
names, inspect label distributions, and map attack types into 5 canonical classes:
Normal, DoS, Probe, R2L, and U2R.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# The 41 standard network traffic features + label and difficulty level in NSL-KDD
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

ALL_COLUMNS = FEATURE_NAMES + ['attack_type', 'difficulty_level']

# Standard 5-class categorization for NSL-KDD attack types
ATTACK_MAPPING = {
    # Normal traffic
    'normal': 'Normal',

    # Denial of Service (DoS)
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'apache2': 'DoS',
    'udpstorm': 'DoS',
    'processtable': 'DoS',
    'worm': 'DoS',
    'mailbomb': 'DoS',

    # Surveillance and Probing (Probe)
    'satan': 'Probe',
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    # Remote to Local (R2L)
    'guess_passwd': 'R2L',
    'ftp_write': 'R2L',
    'imap': 'R2L',
    'phf': 'R2L',
    'multihop': 'R2L',
    'warezmaster': 'R2L',
    'warezclient': 'R2L',
    'spy': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'snmpguess': 'R2L',
    'snmpgetattack': 'R2L',
    'httptunnel': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',

    # User to Root (U2R)
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'rootkit': 'U2R',
    'perl': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R',
    'ps': 'U2R',
}

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def map_attack_label(raw_label: str) -> str:
    """Clean and map an individual attack string to one of the 5 classes."""
    clean_label = str(raw_label).strip().lower().rstrip('.')
    if clean_label in ATTACK_MAPPING:
        return ATTACK_MAPPING[clean_label]
    raise ValueError(
        f"Unknown attack label '{raw_label}'. Please inspect dataset or update ATTACK_MAPPING."
    )


def load_nsl_kdd_file(file_path: Path) -> pd.DataFrame:
    """Load a single NSL-KDD CSV/TXT file and validate its schema.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file does not have expected columns.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found at '{path.resolve()}'.\n"
            f"Please ensure 'KDDTrain+.txt' and 'KDDTest+.txt' are placed inside the 'data/' folder.\n"
            f"Refer to data/README.md for instructions."
        )

    # NSL-KDD records are comma-delimited with no header row
    df = pd.read_csv(path, header=None)

    if df.shape[1] == 43:
        df.columns = ALL_COLUMNS
    elif df.shape[1] == 42:
        # Some subsets omit the difficulty score
        df.columns = FEATURE_NAMES + ['attack_type']
        df['difficulty_level'] = 0
    else:
        raise ValueError(
            f"Unexpected number of columns ({df.shape[1]}) in '{path.name}'. "
            f"Expected 42 or 43 columns."
        )

    # Map raw attack labels to 5 canonical categories
    df['label_5class'] = df['attack_type'].apply(map_attack_label)
    return df


def load_datasets(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both KDDTrain+.txt and KDDTest+.txt from the data directory.

    Returns:
        (train_df, test_df) tuple of pandas DataFrames.
    """
    if data_dir is None:
        # Default to repo root / data
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / 'data'
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / 'KDDTrain+.txt'
    test_path = data_dir / 'KDDTest+.txt'

    train_df = load_nsl_kdd_file(train_path)
    test_df = load_nsl_kdd_file(test_path)

    return train_df, test_df
