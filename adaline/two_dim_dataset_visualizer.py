# Convenience function to vizualize the decision boundaries for 
# two-dimensional datasets

# Extended explanation:
#   define a number of colors and markers and create a colormap from the list of colors via
#   ListedColormap. Then, we determine the minimum and maximum values for the two features and
#   use those feature vectors to create a pair of grid arrays, xx1 and xx2, via the NumPy meshgrid function.
#   Since we trained our perceptron classifier on two feature dimensions, we need to flatten the grid arrays
#   and create a matrix that has the same number of columns as the Iris training subset so that we can
#   use the predict method to predict the class labels, lab, of the corresponding grid points.
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contour(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0],
                    y=X[y == c1, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {c1}',
                    edgecolors='black')