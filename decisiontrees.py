import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from z3 import *
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm  # For progress bar

def extract_decision_path(tree, feature_names):
    """Extract all decision paths from a trained decision tree"""
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    
    def recurse(node, path, paths):
        if children_left[node] == children_right[node]:
            paths.append((path, np.argmax(value[node])))
            return
        
        left_path = path + [(feature_names[feature[node]], threshold[node], "≤")]
        recurse(children_left[node], left_path, paths)
        
        right_path = path + [(feature_names[feature[node]], threshold[node], ">")]
        recurse(children_right[node], right_path, paths)
        
    paths = []
    recurse(0, [], paths)
    return paths

def find_adversarial_example(tree, x_original, target_class, feature_names, feature_bounds, epsilon):
    """
    Find an adversarial example using Z3 solver with specified epsilon constraint
    
    Args:
        epsilon: Maximum allowed L2 distance from original example
    Returns:
        adversarial example if found, None otherwise
    """
    s = Solver()
    
    # Create Z3 variables for features
    z3_vars = {}
    for feature in feature_names:
        z3_vars[feature] = Real(feature)
    
    # Add dataset bounds constraints
    for feature, (lower, upper) in feature_bounds.items():
        s.add(z3_vars[feature] >= lower)
        s.add(z3_vars[feature] <= upper)
    
    # Extract decision paths
    paths = extract_decision_path(tree, feature_names)
    target_paths = [(path, label) for path, label in paths if label == target_class]
    
    # Add path constraints
    path_constraints = []
    for path, _ in target_paths:
        path_constraint = []
        for feature, threshold, op in path:
            if op == "≤":
                path_constraint.append(z3_vars[feature] <= threshold)
            else:
                path_constraint.append(z3_vars[feature] > threshold)
        path_constraints.append(And(path_constraint))
    
    s.add(Or(path_constraints))
    
    # Add L2 distance constraint using epsilon
    # (x1-x1_orig)^2 + (x2-x2_orig)^2 + ... <= epsilon^2
    distance_terms = []
    for i, feature in enumerate(feature_names):
        term = (z3_vars[feature] - x_original[i]) * (z3_vars[feature] - x_original[i])
        distance_terms.append(term)
    
    s.add(Sum(distance_terms) <= epsilon * epsilon)
    
    if s.check() == sat:
        model = s.model()
        adversarial = [float(model[z3_vars[feature]].as_decimal(10))
                      for feature in feature_names]
        return adversarial
    return None

def find_minimal_adversarial(tree, x_original, target_class, feature_names, feature_bounds, 
                           epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9, max_iters=100):
    """
    Iteratively search for the adversarial example with minimal perturbation
    
    Args:
        epsilon_start: Starting epsilon value
        epsilon_min: Minimum epsilon to try
        epsilon_decay: Factor to multiply epsilon by each iteration
        max_iters: Maximum number of iterations
    """
    best_adversarial = None
    best_epsilon = float('inf')
    epsilon = epsilon_start
    
    # Create progress bar
    pbar = tqdm(total=max_iters, desc="Searching for minimal adversarial")
    
    while epsilon >= epsilon_min and pbar.n < max_iters:
        adversarial = find_adversarial_example(
            tree, x_original, target_class, feature_names, feature_bounds, epsilon
        )
        
        if adversarial is not None:
            # Calculate actual L2 distance
            actual_distance = np.linalg.norm(np.array(adversarial) - x_original)
            
            if actual_distance < best_epsilon:
                best_adversarial = adversarial
                best_epsilon = actual_distance
            
            # Reduce epsilon to try finding even smaller perturbation
            epsilon *= epsilon_decay
        else:
            # If no solution found, increase epsilon slightly and try again
            epsilon /= epsilon_decay ** 0.5
        
        pbar.update(1)
        pbar.set_postfix({'best_epsilon': f'{best_epsilon:.4f}', 'current_epsilon': f'{epsilon:.4f}'})
    
    pbar.close()
    return best_adversarial, best_epsilon

def plot_adversarial_search_results(X, y, x_original, adversarial_examples, feature_names, class_names):
    """Plot the original point and all found adversarial examples"""
    feature_pairs = list(combinations(range(len(feature_names)), 2))
    n_pairs = len(feature_pairs)
    n_rows = (n_pairs + 1) // 2
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    for idx, (f1, f2) in enumerate(feature_pairs, 1):
        ax = fig.add_subplot(n_rows, 2, idx)
        
        # Plot dataset points
        for class_idx in np.unique(y):
            mask = y == class_idx
            ax.scatter(X[mask, f1], X[mask, f2], 
                      alpha=0.2, label=f'Class {class_names[class_idx]}')
        
        # Plot original point
        ax.scatter(x_original[f1], x_original[f2], 
                  color='black', marker='*', s=200, label='Original')
        
        # Plot adversarial examples
        for i, adv in enumerate(adversarial_examples):
            ax.scatter(adv[f1], adv[f2], 
                      color='red', alpha=0.5, marker='x', s=100,
                      label=f'Adversarial {i+1}')
        
        ax.set_xlabel(feature_names[f1])
        ax.set_ylabel(feature_names[f2])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Choose example to attack
    x_original = X_test[0]
    original_class = clf.predict([x_original])[0]
    target_class = (original_class + 1) % 3

    # Define feature bounds
    feature_bounds = {}
    for i, feature in enumerate(feature_names):
        feature_bounds[feature] = (X[:, i].min(), X[:, i].max())

    # Search for minimal adversarial example
    best_adversarial, best_epsilon = find_minimal_adversarial(
        clf, x_original, target_class, feature_names, feature_bounds,
        epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9, max_iters=100
    )

    if best_adversarial is not None:
        print("\nOriginal example:")
        for feature, value in zip(feature_names, x_original):
            print(f"{feature}: {value:.4f}")
        print(f"Original class: {class_names[original_class]}")

        print("\nBest adversarial example found:")
        for feature, value in zip(feature_names, best_adversarial):
            print(f"{feature}: {value:.4f}")
        print(f"Adversarial class: {class_names[target_class]}")
        print(f"Minimal L2 distance found: {best_epsilon:.4f}")

        # Calculate and display feature-wise changes
        changes = np.array(best_adversarial) - x_original
        print("\nFeature-wise changes:")
        for feature, change in zip(feature_names, changes):
            print(f"{feature}: {change:.4f}")

        # Plot results
        plot_adversarial_search_results(
            X, y, x_original, [best_adversarial], 
            feature_names, class_names
        )
    else:
        print("No adversarial example found")

if __name__ == "__main__":
    main()
