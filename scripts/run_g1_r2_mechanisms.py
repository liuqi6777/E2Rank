#!/usr/bin/env python3
"""Optional policy, update-rule and frozen-candidate-rescaling ablations."""
from run_g1_r2_ablations import cli

if __name__ == '__main__':
    cli(('query_policy', 'document_policy', 'norm', 'doc_mean', 'norm_doc_mean',
         'no_rescale_sf', 'no_rescale_cp'))
