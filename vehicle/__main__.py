import argparse
from .driver import main

parser = argparse.ArgumentParser(description='Vehicle model with 3DOF.')
parser.add_argument("--parameters")
parser.add_argument("--telemetry")
args = parser.parse_args()

main()