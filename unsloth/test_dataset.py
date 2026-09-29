import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset

# Try loading the datasets
try:
    ds1 = load_dataset('CrowdMind/sft-tool-use-collection', split='train')
    print('Dataset 1 loaded:', ds1.column_names)
    print('First row keys:', ds1[0].keys())
    print('First row data:', ds1[0])
except Exception as e:
    print('Dataset 1 error:', type(e).__name__, e)

try:
    ds2 = load_dataset('CrowdMind/sft-other', split='train')
    print('Dataset 2 loaded:', ds2.column_names)
    print('First row keys:', ds2[0].keys())
    print('First row data:', ds2[0])
except Exception as e:
    print('Dataset 2 error:', type(e).__name__, e)