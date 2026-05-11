# Per-dataset BCE loss weights for edge-type classification (11 classes).
# Each dataset has a manually tuned 11-element list [w_0, ..., w_10] where
# class 10 is always 0 (self-loops, not used in evaluation).
# These are used as the `weight` argument to nn.BCEWithLogitsLoss during
# edge-type decoder fine-tuning in link_prediction().

DATASET_WEIGHTS = {
    "THEIA_E3": [
        4.45,   # class 0
        10.66,  # class 1
        1.14,   # class 2
        1.31,   # class 3
        5.08,   # class 4
        17.45,  # class 5
        17.15,  # class 6
        4.82,   # class 7
        5.09,   # class 8
        7.71,   # class 9
        0.0,    # class 10 (self-loop, no samples)
    ],
    "THEIA_E3_change_validation_1": [
        3.75,   # class 0
        9.95,   # class 1
        1.17,   # class 2
        1.52,   # class 3
        4.27,   # class 4
        16.60,  # class 5
        21.81,  # class 6
        4.10,   # class 7
        4.36,   # class 8
        5.49,   # class 9
        0.0,    # class 10 (self-loop, no samples)
    ],
    "THEIA_E3_change_validation_2": [
        4.27,   # class 0
        6.99,   # class 1
        1.14,   # class 2
        1.42,   # class 3
        4.76,   # class 4
        18.71,  # class 5
        22.48,  # class 6
        4.63,   # class 7
        4.58,   # class 8
        6.13,   # class 9
        0.0,    # class 10 (self-loop, no samples)
    ],
    "THEIA_E3_change_validation_3": [
        3.97,   # class 0
        8.02,   # class 1
        1.16,   # class 2
        1.45,   # class 3
        4.51,   # class 4
        18.34,  # class 5
        21.50,  # class 6
        4.41,   # class 7
        4.36,   # class 8
        6.11,   # class 9
        0.0,    # class 10 (self-loop, no samples)
    ],
    "CADETS_E3": [
        6.67,   # class 0
        5.06,   # class 1
        0.85,   # class 2
        1.53,   # class 3
        9.26,   # class 4
        30.0,   # class 5 (capped)
        30.0,   # class 6 (capped)
        8.39,   # class 7
        5.11,   # class 8
        0.0,    # class 9 (no samples)
        0.0,    # class 10 (self-loop, no samples)
    ],
    "CLEARSCOPE_E3": [
        4.58,   # class 0
        0.0,    # class 1 (no samples)
        1.12,   # class 2
        1.91,   # class 3
        4.37,   # class 4
        30.0,   # class 5 (capped)
        30.0,   # class 6 (capped)
        4.04,   # class 7
        2.09,   # class 8
        0.0,    # class 9 (no samples)
        0.0,    # class 10 (self-loop, no samples)
    ],
    # THEIA_E5: using the normalized-graph variant
    "THEIA_E5": [
        24.84,  # class 0
        4.82,   # class 1
        0.69,   # class 2
        1.94,   # class 3
        30.0,   # class 4 (capped)
        30.0,   # class 5 (capped)
        30.0,   # class 6 (capped)
        22.91,  # class 7
        8.01,   # class 8
        4.24,   # class 9
        0.0,    # class 10 (self-loop, no samples)
    ],
    "CLEARSCOPE_E5": [
        11.23,  # class 0
        30.0,   # class 1 (capped)
        0.88,   # class 2
        1.25,   # class 3
        15.0,   # class 4 (capped)
        15.0,   # class 5 (capped)
        15.0,   # class 6 (capped)
        15.0,   # class 7 (capped)
        5.89,   # class 8
        30.0,   # class 9 (capped)
        0.0,    # class 10 (self-loop, no samples)
    ],
    # CADETS_E5: using the normalized-graph variant
    "CADETS_E5": [
        22.48,  # class 0
        4.10,   # class 1
        0.73,   # class 2
        1.78,   # class 3
        30.0,   # class 4 (capped)
        30.0,   # class 5 (capped)
        30.0,   # class 6 (capped)
        22.70,  # class 7
        4.33,   # class 8
        0.0,    # class 9 (no samples)
        0.0,    # class 10 (self-loop, no samples)
    ],
    # OPTC_h201: using the merged-edges variant
    "OPTC_h201": [
        4.53,   # class 0
        3.15,   # class 1
        7.97,   # class 2
        1.56,   # class 3
        6.45,   # class 4
        0.97,   # class 5
        23.60,  # class 6
        15.56,  # class 7
        0.0,    # class 8 (no samples)
        9.29,   # class 9
        0.0,    # class 10 (self-loop, no samples)
    ],
    "OPTC_h051": [
        5.05,   # class 0
        3.31,   # class 1
        8.66,   # class 2
        1.53,   # class 3
        5.76,   # class 4
        0.96,   # class 5
        23.84,  # class 6
        15.77,  # class 7
        0.0,    # class 8 (no samples)
        9.59,   # class 9
        0.0,    # class 10 (self-loop, no samples)
    ],
    "OPTC_h501": [
        4.43,   # class 0
        3.11,   # class 1
        7.73,   # class 2
        1.57,   # class 3
        6.23,   # class 4
        0.98,   # class 5
        23.73,  # class 6
        15.81,  # class 7
        0.0,    # class 8 (no samples)
        9.47,   # class 9
        0.0,    # class 10 (self-loop, no samples)
    ],
}
