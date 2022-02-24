import kfold_template
import pandas
from matplotlib import pyplot
from sklearn import svm

#SVM: Support Vector Machine
dataset = pandas.read_csv("dataset_svm_1.csv")

target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:3].values

pyplot.scatter(data[:, 0], data[:, 1], c = target)
pyplot.savefig("scatterplot1.png")
pyplot.close()

#looking at the plot you can see that there's a overlap between yellow and purple points

r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, target, 4, svm.SVC(kernel = "linear"), 1, 1)

print(r2_scores)
print(accuracy_scores)

#fix the confusion matrices