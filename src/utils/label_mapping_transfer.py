label_to_id = {
    'O': 0,
    'B-PROD': 1,
    'I-PROD': 2,
    'B-FELT': 3,
    'I-FELT': 4,
    'B-EVT': 5,
    'I-EVT': 6,
    'B-GPE_LOC': 7,
    'I-GPE_LOC': 8,
    'B-PER': 9,
    'I-PER': 10,
    'B-MISC': 11,
    'I-MISC': 12,
    'B-GPE_ORG': 13,
    'I-GPE_ORG': 14,
    'B-ORG': 15,
    'I-ORG': 16,
    'B-DRV': 17,
    'I-DRV': 18
}

id_to_label = {v: k for k, v in label_to_id.items()}