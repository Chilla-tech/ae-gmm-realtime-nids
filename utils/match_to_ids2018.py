# Standardize column names to CIC-IDS2018, then get intersecting features
import re
import pandas as pd

# 1) Canonical CIC-IDS2018 names (your feature_naames_IDS2018.txt)
IDS2018_CANON = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
       'Timestamp', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
       'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
       'Fwd Packet Length Max', 'Fwd Packet Length Min',
       'Fwd Packet Length Mean', 'Fwd Packet Length Std',
       'Bwd Packet Length Max', 'Bwd Packet Length Min',
       'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
       'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
       'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
       'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
       'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
       'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd RST Flags',
       'Bwd RST Flags', 'Fwd Header Length', 'Bwd Header Length',
       'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min',
       'Packet Length Max', 'Packet Length Mean', 'Packet Length Std',
       'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
       'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
       'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
       'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
       'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
       'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
       'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
       'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
       'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
       'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
       'Idle Min', 'ICMP Code', 'ICMP Type', 'Total TCP Flow Time', 'Label']

CANON_SET = set(IDS2018_CANON)

# 2) Known aliases and abbreviations -> canonical (extend as needed)
ALIASES = {
    # Spacing/camel-case and expanded names
    'SrcIP': 'Src IP',
    'Source IP': 'Src IP',
    'DstIP': 'Dst IP',
    'Destination IP': 'Dst IP',
    'SrcPort': 'Src Port',
    'Source Port': 'Src Port',
    'DstPort': 'Dst Port',
    'Destination Port': 'Dst Port',
    'Timestamp': 'Timestamp',
    'Protocol': 'Protocol',

    # Totals and lengths
    'Tot Fwd Pkts': 'Total Fwd Packet',
    'Total Fwd Packets': 'Total Fwd Packet',
    'Tot Bwd Pkts': 'Total Bwd packets',
    'Total Backward Packets': 'Total Bwd packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packet',
    'Total Length of Fwd Packets': 'Total Length of Fwd Packet',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packet',
    'Total Length of Bwd Packets': 'Total Length of Bwd Packet',
    'Total Connection Flow Time': 'Total TCP Flow Time',

    # Packet length stats
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min',
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': 'Bwd Packet Length Min',
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std',
    'Max Packet Length': 'Packet Length Max',
    'Min Packet Length': 'Packet Length Min',
    'Pkt Len Min': 'Packet Length Min',
    'Pkt Len Max': 'Packet Length Max',
    'Pkt Len Mean': 'Packet Length Mean',
    'Pkt Len Std': 'Packet Length Std',
    'Pkt Len Var': 'Packet Length Variance',
    'Pkt Size Avg': 'Average Packet Size',

    # IAT stats
    'Flow IAT Stddev': 'Flow IAT Std',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Bwd IAT Tot': 'Bwd IAT Total',

    # Flags abbreviations/case
    'Fwd PSH flags': 'Fwd PSH Flags',
    'Bwd PSH flags': 'Bwd PSH Flags',
    'Fwd URG flags': 'Fwd URG Flags',
    'Bwd URG flags': 'Bwd URG Flags',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
    'CWR Flag Cnt': 'CWR Flag Count',
    'CWE Flag Count': 'CWR Flag Count',  # Typo fix
    'ECE Flag Cnt': 'ECE Flag Count',

    # Rates and averages
    'Flow Bytes/s': 'Flow Bytes/s',
    'Flow Packets/s': 'Flow Packets/s',
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': 'Bwd Packets/s',
    'Avg Pkt Size': 'Average Packet Size',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': 'Flow Packets/s',

    # Segment sizes (multiple variants)
    'Fwd Seg Size Avg': 'Fwd Segment Size Avg',
    'Avg Fwd Segment Size': 'Fwd Segment Size Avg',
    'Bwd Seg Size Avg': 'Bwd Segment Size Avg',
    'Avg Bwd Segment Size': 'Bwd Segment Size Avg',
    
    # Segment size minimum
    'min_seg_size_forward': 'Fwd Seg Size Min',
    'Min Seg Size Forward': 'Fwd Seg Size Min',
    'Backward Segment Size Minimum': 'Bwd Seg Size Min',
    'Min Seg Size Backward': 'Bwd Seg Size Min',

    # Bulk rate variants
    'Fwd Avg Bulk Rate': 'Fwd Bulk Rate Avg',
    'Bwd Avg Bulk Rate': 'Bwd Bulk Rate Avg',
    'Fwd Avg Bytes/Bulk': 'Fwd Bytes/Bulk Avg',
    'Bwd Avg Bytes/Bulk': 'Bwd Bytes/Bulk Avg',
    'Fwd Avg Packets/Bulk': 'Fwd Packet/Bulk Avg',
    'Bwd Avg Packets/Bulk': 'Bwd Packet/Bulk Avg',
    'Fwd Byts/b Avg': 'Fwd Bytes/Bulk Avg',
    'Fwd Pkts/b Avg': 'Fwd Packet/Bulk Avg',
    'Fwd Blk Rate Avg': 'Fwd Bulk Rate Avg',
    'Bwd Byts/b Avg': 'Bwd Bytes/Bulk Avg',
    'Bwd Pkts/b Avg': 'Bwd Packet/Bulk Avg',
    'Bwd Blk Rate Avg': 'Bwd Bulk Rate Avg',

    # Subflow
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Bwd Pkts': 'Subflow Bwd Packets',

    # Init window bytes (case variants and underscore versions)
    'Fwd Init Win Bytes': 'FWD Init Win Bytes',
    'Init Fwd Win Bytes': 'FWD Init Win Bytes',
    'Init Fwd Win Byts': 'FWD Init Win Bytes',
    'Init_Win_bytes_forward': 'FWD Init Win Bytes',
    'Init Bwd Win Bytes': 'Bwd Init Win Bytes',
    'Init Bwd Win Byts': 'Bwd Init Win Bytes',
    'Init_Win_bytes_backward': 'Bwd Init Win Bytes',

    # Header length variants
    'Fwd Header Length.1': 'Fwd Header Length',
    
    # Active data packets (underscore and mixed case variants)
    'act_data_pkt_fwd': 'Fwd Act Data Pkts',
    'Act Data Pkt Fwd': 'Fwd Act Data Pkts',
    'act_data_pkt_bwd': 'Bwd Act Data Pkts',
    'Act Data Pkt Bwd': 'Bwd Act Data Pkts',
    
    # Retransmission variants
    #'Fwd TCP Retrans. Count': 'Fwd Act Data Pkts',  # Approximate mapping
    #'Bwd TCP Retrans. Count': 'Bwd Act Data Pkts',  # Approximate mapping
    #'Total TCP Retrans. Count': 'Total TCP Flow Time',  # Approximate mapping

    # Other field variants
    #'SimillarHTTP': 'SimilarHTTP',  # Typo fix
    #'Inbound': 'Direction',  # Directional indicator

    # Label
    'Attack': 'Label'
}

# 3) Normalizer for fuzzy matching
def _norm(s: str) -> str:
    s = str(s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = (s.replace('Cnt', 'Count')
           .replace('Pkts', 'Packets')
           .replace('Pkt', 'Packet')
           .replace('Stddev', 'Std')
           .replace('Bytes/s', 'Bytes per s')
           .replace('Packets/s', 'Packets per s'))
    return re.sub(r'[^a-z0-9]', '', s.lower())

CANON_NORM = { _norm(c): c for c in IDS2018_CANON }

def standardize_to_ids2018(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where columns are renamed to CIC-IDS2018 canonical names when possible."""
    cols = pd.Index(df.columns).map(lambda s: re.sub(r'\s+', ' ', str(s)).strip())
    df2 = df.copy()
    df2.columns = cols

    rename = {}

    # First pass: direct alias map
    for c in df2.columns:
        if c in ALIASES:
            rename[c] = ALIASES[c]

    # Second pass: normalized fuzzy match to canonical
    for c in df2.columns:
        if c in rename:
            continue
        n = _norm(c)
        if n in CANON_NORM:
            rename[c] = CANON_NORM[n]

    if rename:
        df2 = df2.rename(columns=rename)

    return df2



features_dos=['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
       'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packet',
       'Total Bwd packets', 'Total Length of Fwd Packet',
       'Total Length of Bwd Packet', 'Fwd Packet Length Max',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean',
       'Fwd Packet Length Std', 'Bwd Packet Length Max',
       'Bwd Packet Length Min', 'Bwd Packet Length Mean',
       'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
       'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
       'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
       'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
       'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
       'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
       'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
       'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
       'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
       'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
       'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
       'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg','Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg',
       'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg',
       'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
       'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes',
       'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
       'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
       'Idle Std', 'Idle Max', 'Idle Min', 'Label']


def ddos_prep_n_sample(dataset_list, benign_set, features=features_dos):
    mergeData=benign_set[features]
    for dataset, info in dataset_list.items():
        data_name=dataset
        print(f'Preparing dataset {data_name}')
        num=info['num_samples']
        print(f'{num} of samples will be selected')
        data_name=pd.read_csv(info['path_to'])
        data_name=data_name.drop(columns=[' Fwd Header Length.1'])
        data_name=standardize_to_ids2018(data_name)
        data_name=data_name[features]
        data_name=data_name[data_name['Label']!='BENIGN']
        mergeData=pd.concat([mergeData, data_name[:num]], ignore_index=True)
        #print(f'Dataset Distribution after merging with {dataset}\n{mergeData['Label'].value_counts()}')
    return mergeData