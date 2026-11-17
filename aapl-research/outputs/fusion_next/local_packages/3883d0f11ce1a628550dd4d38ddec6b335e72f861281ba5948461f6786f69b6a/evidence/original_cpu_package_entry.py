"""Use the already adapted canonical package module with its original CLI."""
import argparse
from pathlib import Path

from fusion.next import local_package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    archive = local_package.build(args.deployment)
    print(local_package.verify(archive) if args.verify else archive)


if __name__ == '__main__':
    main()
