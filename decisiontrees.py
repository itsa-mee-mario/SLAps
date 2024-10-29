import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from z3 import *
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_adversarial_example(X, y, x_original, adversarial, feature_names, class_names, epsilon=1.0):
    """
    Create multiple plots showing the adversarial example in different feature spaces
    """
    # Get all possible pairs of features
    feature_pairs = list(combinations(range(len(feature_names)), 2))
    n_pairs = len(feature_pairs)

    # Set up the plotting grid
    n_rows = (n_pairs + 1) // 2
    fig = plt.figure(figsize=(15, 5 * n_rows))

    # Create a subplot for each feature pair
    for idx, (f1, f2) in enumerate(feature_pairs, 1):
        ax = fig.add_subplot(n_rows, 2, idx)

        # Plot original data points
        for class_idx in np.unique(y):
            mask = y == class_idx
            ax.scatter(X[mask, f1], X[mask, f2],
                      alpha=0.5, label=f'Class {class_names[class_idx]}')

        # Plot original point
        ax.scatter(x_original[f1], x_original[f2],
                  color='black', marker='*', s=200, label='Original')

        # Plot adversarial example
        ax.scatter(adversarial[f1], adversarial[f2],
                  color='red', marker='X', s=200, label='Adversarial')

        # Plot epsilon ball
        circle = plt.Circle((x_original[f1], x_original[f2]), epsilon,
                          color='gray', fill=False, linestyle='--', label='ε-ball')
        ax.add_artist(circle)

        # Add labels and legend
        ax.set_xlabel(feature_names[f1])
        ax.set_ylabel(feature_names[f2])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

    plt.tight_layout()

    # Create 3D plot using first three features
    fig3d = plt.figure(figsize=(10, 10))
    ax3d = fig3d.add_subplot(111, projection='3d')

    # Plot original data points in 3D
    for class_idx in np.unique(y):
        mask = y == class_idx
        ax3d.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                    alpha=0.5, label=f'Class {class_names[class_idx]}')

    # Plot original point and adversarial example in 3D
    ax3d.scatter(x_original[0], x_original[1], x_original[2],
                color='black', marker='*', s=200, label='Original')
    ax3d.scatter(adversarial[0], adversarial[1], adversarial[2],
                color='red', marker='X', s=200, label='Adversarial')

    # Add labels and legend
    ax3d.set_xlabel(feature_names[0])
    ax3d.set_ylabel(feature_names[1])
    ax3d.set_zlabel(feature_names[2])
    ax3d.legend()

    plt.show()

    # Create a plot showing the feature-wise changes
    fig, ax = plt.figure(figsize=(10, 5)), plt.axes()

    x = np.arange(len(feature_names))
    width = 0.35

    # Plot original and adversarial values side by side
    ax.bar(x - width/2, x_original, width, label='Original', color='blue', alpha=0.5)
    ax.bar(x + width/2, adversarial, width, label='Adversarial', color='red', alpha=0.5)

    # Add feature names and labels
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature-wise Comparison')
    ax.legend()

    # Add arrows showing the perturbation
    for i in range(len(feature_names)):
        plt.arrow(i, x_original[i], 0, adversarial[i] - x_original[i],
                 head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.5)

    plt.tight_layout()
    plt.show()

# Load and prepare data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Train model and find adversarial example (using code from previous artifact)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Choose a test example
x_original = X_test[0]
original_class = clf.predict([x_original])[0]
target_class = (original_class + 1) % 3

# Define feature bounds
feature_bounds = {}
for i, feature in enumerate(feature_names):
    feature_bounds[feature] = (X[:, i].min(), X[:, i].max())

# Find adversarial example using the function from previous artifact
adversarial = find_adversarial_example(
    clf, x_original, target_class, feature_names, feature_bounds
)

if adversarial:
    # Calculate epsilon (L2 norm of perturbation)
    epsilon = np.linalg.norm(np.array(adversarial) - x_original)

    # Create visualizations
    plot_adversarial_example(X, y, x_original, adversarial,
                           feature_names, class_names, epsilon)

    # Print numerical results
    print("\nNumerical Results:")
    print(f"Original class: {class_names[original_class]}")
    print(f"Adversarial class: {class_names[target_class]}")
    print(f"Epsilon (L2 norm of perturbation): {epsilon:.4f}")

    # Print feature-wise changes
    print("\nFeature-wise changes:")
    for i, feature in enumerate(feature_names):
        change = adversarial[i] - x_original[i]
        print(f"{feature}: {x_original[i]:.2f} -> {adversarial[i]:.2f} (Δ = {change:.2f})")
else:
    print("No adversarial example found")
