import argparse
import multiprocessing


def get_options():
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='MiniProject 3: Modified MNIST. Authors: Ramon Figueiredo Pessoa, Yujing Zou, Rui Ma',
                                     epilog='COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.')
    parser.add_argument("-d", "--dataset",
                        action="store", dest="dataset",
                        help="Dataset used (Options: MNIST). Default: MNIST",
                        default='MNIST')
    parser.add_argument("-n_jobs",
                        action="store", type=int, dest="n_jobs", default=-1,
                        help="The number of CPUs to use to do the computation. "
                             "If the provided number is negative or greater than the number of available CPUs, "
                             "the system will use all the available CPUs. Default: -1 (-1 == all CPUs)")
    parser.add_argument('-save_logs', '--save_logs_in_file', action='store_true', default=False,
                        dest='save_logs_in_file',
                        help='Save logs in a file. Default: False (show logs in the prompt)')
    parser.add_argument('-verbose', '--verbosity', action='store_true', default=False,
                        dest='verbose',
                        help='Increase output verbosity. Default: False')
    parser.add_argument("-random_state",
                        action="store", type=int, dest="random_state", default=0,
                        help="Seed used by the random number generator. Default: 0")
    parser.add_argument('-v', '--version', action='version', dest='version', version='%(prog)s 1.0')

    options = parser.parse_args()

    if options.n_jobs > multiprocessing.cpu_count() or (options.n_jobs != -1 and options.n_jobs < 1):
        options.n_jobs = -1  # use all available cpus

    options.dataset = options.dataset.upper().strip()

    show_option(options, parser)

    return options


def show_option(options, parser):

    print('=' * 130)

    print('\nRunning with options: ')
    print('\tDataset =', options.dataset)
    print('\tThe number of CPUs to use to do the computation. Default: -1 (-1 == all CPUs) =', options.n_jobs)
    print('\tSave logs in a file. Default: False (show logs in the prompt) =', options.save_logs_in_file)
    print('\tVerbose =', options.verbose)
    print('\tSeed used by the random number generator (random_state) =', options.random_state)
    print('=' * 130)
    print()
