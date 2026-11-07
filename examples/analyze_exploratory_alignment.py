"""EXPLORATORY / NOT VALIDATED metadata-only EDA; never runs a baseline."""
import argparse
import json

from bd3d_quality.exploratory_alignment import analyze


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('output', help='New directory; existing outputs are never overwritten')
    args = parser.parse_args()
    print(json.dumps(analyze(args.bundle, args.output)))
