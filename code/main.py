import logging
import os
from time import time

from argument_parser.argument_parser import get_options

if __name__ == '__main__':
	options = get_options()

	if options.save_logs_in_file:
		if not os.path.exists('logs'):
			os.mkdir('logs')
		logging.basicConfig(filename='logs/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
							level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
	else:
		logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
							datefmt='%m/%d/%Y %I:%M:%S %p')

	logging.info("Program started...")

	start = time()

	print('\n\nDONE!')

	print("Program finished. It took {} seconds".format(time() - start))
