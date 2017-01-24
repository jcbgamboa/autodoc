import argparse
import os

import numpy as np
from sklearn.metrics import confusion_matrix
import pandas

def get_accuracies(n_runs, results_path):
	accuracy_file_path = os.path.join('cnn', 'accuracy.csv')

	#corrects_list = []
	#totals_list = []
	accuracy_list = []
	for i in range(n_runs):
		dir_path = os.path.join(results_path, 'run_' + str(i))
		curr_accuracy_file = os.path.join(dir_path, accuracy_file_path)

		with open(curr_accuracy_file, 'r') as f:
			line = f.readline()
			correct, total = line.split(',')

			#corrects_list.append(correct)
			#totals_list.append(total)
			accuracy_list.append(float(correct) / float(total))

	#return corrects_list, totals_list, accuracy_list
	return accuracy_list

def calculate_final_accuracy(n_runs, results_path):
	accuracy_list = get_accuracies(n_runs, results_path)

	#assert(len(corrects_list) == len(totals_list) == len(accuracy_list))

	accuracies = np.array(accuracy_list)

	accuracy_mean = accuracies.mean()
	accuracy_variance = accuracies.var()
	accuracy_median = np.median(accuracies)

	#print("Accuracies: {}".format(accuracies))

	return accuracy_mean, accuracy_variance, accuracy_median

def get_confusion_matrices(n_runs, results_path):
	test_metrics_file_path = os.path.join('cnn', 'test_metrics.csv')

	confusion_matrices = []
	for i in range(n_runs):
		dir_path = os.path.join(results_path, 'run_' + str(i))
		curr_test_metrics_file = os.path.join(dir_path,
							test_metrics_file_path)

		with open(curr_test_metrics_file, 'r') as f:
			csv = pandas.read_csv(curr_test_metrics_file,
							header = None)

			confusion_matrices.append(confusion_matrix(
							csv[1], csv[0]))

	return confusion_matrices

def calculate_final_confusion_matrix(n_runs, results_path):
	confusion_matrices = get_confusion_matrices(n_runs, results_path)

	#print("Confusion Matrices:")
	#for i in confusion_matrices:
	#	print(i)

	final_cm = np.array(confusion_matrices)

	final_cm_means = final_cm.mean(axis = 0)
	final_cm_variances = final_cm.var(axis = 0)
	final_cm_medians = np.median(final_cm, axis = 0)

	return final_cm, final_cm_means, final_cm_variances, final_cm_medians


def dump_results(acc_mean, acc_var, acc_median,
		cms, cm_means, cm_vars, cm_medians):
	# For now, just print to the screen

	print("Accuracy mean: {}".format(acc_mean))
	print("Accuracy variance: {}".format(acc_var))
	print("Accuracy median: {}".format(acc_median))

	# Not printing each confusion matrix
	# for i in range(cms.shape[0]):
	#	...

	print("Confusion Matrix -- Means")
	print(cm_means)

	print("Confusion Matrix -- Variances")
	print(cm_vars)

	print("Confusion Matrix -- Medians")
	print(cm_medians)

def main():
	args = parse_command_line()

	acc_mean, acc_var, acc_median = calculate_final_accuracy(
					args.n_runs, args.results_path)

	cms, cm_means, cm_vars, cm_medians = calculate_final_confusion_matrix(
					args.n_runs, args.results_path)

	dump_results(acc_mean, acc_var, acc_median,
			cms, cm_means, cm_vars, cm_medians)

def parse_command_line():
	parser = argparse.ArgumentParser(
		description='Given a path with the results of many runs of' +
			'a network, generate accuracy and a Confusion Matrix' +
			'based on all of them.')

	parser.add_argument('results_path', metavar = 'results_path',
		type = str,
		help = 'Path to the root of "where the results are".')

	parser.add_argument('n_runs', metavar = 'n_runs', type = int,
		help = 'Number of runs of the model.')

	parser.add_argument('output_path', metavar = 'output_path', type = str,
		help = 'Where should the output be put. A file called ' +
		'"accuracy.txt" will be created, as well as another called ' +
		'"confusion_matrix.txt".')

	return parser.parse_args()

if __name__ == '__main__':
	main()
