#!./python3.6/bin/python

import argparse

from pyrouge import Rouge155

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(help="Path of the directory containing ROUGE-1.5.5.pl.",
	type=str, action="store", dest="home_dir")
	return parser.parse_args()

def main():
	args = get_args()
	Rouge155(args.home_dir)

if __name__ == "__main__":
	main()
