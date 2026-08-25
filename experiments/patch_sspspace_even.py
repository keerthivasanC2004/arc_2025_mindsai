#!/usr/bin/env python3
import argparse
from pathlib import Path

OLD = '''def conjsym(K, even=False):
    d = K.shape[0]
    n = d*2 + 1 #+ 1*even
    F = np.zeros((n,K.shape[1]), dtype="complex")
    # F[1:round(n/2),:] = K
    # F[round(n/2 + 1):,:] = -np.flip(K,axis=0)
    F[1:(d+1),:] = K
    F[(d + 1):,:] = -np.flip(K,axis=0)
    return F.real
'''

NEW = '''def conjsym(K, even=False):
    d = K.shape[0]
    n = d*2 + 1 + int(even)
    F = np.zeros((n,K.shape[1]), dtype="complex")
    F[1:(d+1),:] = K
    neg_start = d + 2 if even else d + 1
    F[neg_start:,:] = -np.flip(K,axis=0)
    return F.real
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    args = ap.parse_args()
    p = Path(args.path)
    src = p.read_text()
    if OLD not in src:
        raise RuntimeError('Expected pinned conjsym implementation not found')
    p.write_text(src.replace(OLD, NEW, 1))
    print('PATCHED_EVEN_CONJSYM', p)


if __name__ == '__main__':
    main()
