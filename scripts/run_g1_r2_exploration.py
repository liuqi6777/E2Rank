#!/usr/bin/env python3
"""Optional broad alignment sweep, annealing and Gaussian-mismatch diagnostic."""
from run_g1_r2_ablations import cli

if __name__ == '__main__':
    cli(tuple(f'align{point}_{est}' for point in ('040', '053', '065', '080', '095', '098')
              for est in ('sf', 'cp')) + ('anneal_sf', 'anneal_cp', 'vmf_k755', 'gaussian_k755'))
