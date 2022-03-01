import kfold_template
import pandas
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import numpy

dataset = pandas.read_csv("dataset_svm_2.csv")

dataset['x3'] = dataset['x1']**2 + dataset['x2']**2
target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:4].values

fig = pyplot.figure()
#111, a one by one plot, the first plot; 221, a two by two plot, the first plot; and so forth...
fig1 = fig.add_subplot(111, projection = '3d')
#matplotlib requires to transform pandas type values into numpy values...
x1 = data[:, 0].reshape(-1, 1)
#x1 is not a vector, but rather a one column matrix...
x2 = data[:, 1].reshape(-1, 1)
x3 = data[:, 2].reshape(-1, 1)

fig1.scatter(x1, x2, x3, c = target, depthshade = True)

machine = svm.SVC(kernel = "linear")
machine.fit(data, target)
coef = machine.coef_
intercept = machine.intercept_
print(coef)
print(intercept)

#output: [[-0.05116273  0.09036613 -8.26713984]][6.77345861]

x1, x2 = numpy.meshgrid(x1, x2)
plane = -(coef[0][0]*x1 + coef[0][1]*x2 + intercept)/coef[0][2]
fig_surface = fig.gca(projection = "3d")
fig_surface.plot_surface(x1, x2, plane, alpha = 0.01)
pyplot.savefig("scatter_3D.png")
pyplot.close()

#you get the plane by equaling the regression to zero and solving for 'x3'
r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, target, 4, svm.SVC(kernel = "linear"), 1, 1)

print(r2_scores)