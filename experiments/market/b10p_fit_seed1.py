"""B10-P fit from the acquired seed-1 native tape model (seed spread for the phase-1 selection); same driver otherwise."""
import sys
import b10p_fit
b10p_fit.ACQUIRED = {'native': 'experiments/results/market/aws_b10_stable_native_s1_20261014_v1.json'}
b10p_fit.SOURCES = b10p_fit.SOURCES + ['experiments/market/b10p_fit_seed1.py']
if __name__ == '__main__':
    b10p_fit.main()
