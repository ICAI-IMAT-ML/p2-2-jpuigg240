# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """

    return (np.sum(np.abs(a - b) ** p)) ** (1 / p)


# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        # Validar que X_train y y_train tengan el mismo número de filas.
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Length of X_train and y_train must be equal.")

        # Validar que k y p sean enteros positivos.
        if not (isinstance(k, int) and k > 0) or not (isinstance(p, int) and p > 0):
            raise ValueError("k and p must be positive integers.")

        # # Validar que p sea un entero positivo.
        # if not (isinstance(p, int) and p > 0):
        #     raise ValueError("p must be a positive integer.")

        self.x_train = X_train
        self.y_train = y_train
        self.k = k
        self.p = p

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predictions = []
        for point in X:
            # Calcula las distancias desde el punto a cada muestra de entrenamiento
            distances = self.compute_distances(point)
            # Ordena las distancias de forma ascendente
            neighbour_ind = self.get_k_nearest_neighbors(distances)
            # Añade a mi modelo de entrenamiento los indices de la distancias
            neighbour_labels = self.y_train[neighbour_ind]
            # Calcular cual es la etiqueta mas comun entre los vecinos
            labels = self.most_common_label(neighbour_labels)
            # Añadimos las etiquetas a nuestra lista de predicciones
            predictions.append(labels)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        proba_list = []
        classes = np.unique(self.y_train)
        
        for point in X:
            # Mismas variables que en la funcion predict excepto que aqui calculamos la prob
            distances = self.compute_distances(point)
            neighbour_ind = self.get_k_nearest_neighbors(distances)
            neighbour_labels = self.y_train[neighbour_ind]
            prob_list = []
            for cls in classes:
                # Obtener un array booleano indicando dónde coincide la etiqueta con la clase actual.
                matches = (neighbour_labels == cls)
                # Sumar los valores True (1) para contar cuántos vecinos tienen esa clase.
                count = np.sum(matches)
                # Dividir por el número total de vecinos (self.k) para obtener la proporción.
                proportion = count / self.k
                prob_list.append(proportion)

            prob = np.array(prob_list)
            proba_list.append(prob)
  

        return np.array(proba_list)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        # Calcular la distancia de Minkowski entre 'point' y cada muestra de entrenamiento.
        distances = np.array([
            np.sum(np.abs(point - train_point) ** self.p) ** (1 / self.p)
            for train_point in self.x_train
        ])
        return distances

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        # Ordenamos los indices de las distancias de menor a mayor hasta ek indice k
        return np.argsort(distances)[:self.k]

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        # Usar np.unique para contar las ocurrencias de cada etiqueta.
        unique_labels, counts = np.unique(knn_labels, return_counts=True)
        # Obtener el índice de la etiqueta con mayor frecuencia.
        max_index = np.argmax(counts)
        return unique_labels[max_index]

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Calcular los componentes de la matriz de confusión:
    # Verdaderos Positivos (TP): y_true == 1 y y_pred == 1
    tp = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
    # Verdaderos Negativos (TN): y_true == 0 y y_pred == 0
    tn = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))
    # Falsos Positivos (FP): y_true == 0 pero y_pred == 1
    fp = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
    # Falsos Negativos (FN): y_true == 1 pero y_pred == 0
    fn = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))

    total = tp + tn + fp + fn

    # Accuracy: (TP + TN) / total
    accuracy = (tp + tn) / total if total != 0 else 0

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0

    # Recall (Sensibilidad): TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0

    # Specificity: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }



def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    y_true_binary = np.array([1 if label == positive_label else 0 for label in y_true])
    
    # Generar los bordes: Se necesitan n_bins+1 puntos para formar n_bins intervalos
    bordes = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bordes[:-1] + bordes[1:]) / 2.0
    
    true_proportions = []
    for i in range(n_bins):
        lower_edge = bordes[i]
        upper_edge = bordes[i + 1]
        
        # Para todos los bins excepto el último: [lower_edge, upper_edge)
        if i < n_bins - 1:
            indices = (y_probs >= lower_edge) & (y_probs < upper_edge)
        else:
            # Para el último bin incluimos el límite superior: [lower_edge, upper_edge]
            indices = (y_probs >= lower_edge) & (y_probs <= upper_edge)
        
        if np.sum(indices) > 0:
            proportion = np.sum(y_true_binary[indices]) / np.sum(indices)
        else:
            proportion = np.nan  # Si el bin está vacío
        true_proportions.append(proportion)
    
    true_proportions = np.array(true_proportions)
    
    # Graficar la curva de calibración
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, true_proportions, marker='o', linestyle='-', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel('Mean predicted probability (bin center)')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
 

    return {"bin_centers": bin_centers, "true_proportions": true_proportions}



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    # Extraer las probabilidades correspondientes a cada clase.
    pos_probs = y_probs[y_true_mapped == 1]
    neg_probs = y_probs[y_true_mapped == 0]

    # Graficar los histogramas.
    plt.figure(figsize=(12, 5))

    # Histograma para la clase positiva.
    plt.subplot(1, 2, 1)
    plt.hist(pos_probs, bins=n_bins, color='blue', edgecolor='black')
    plt.title("Histogram of predicted probabilities\nfor positive class")
    plt.xlabel("Predicted probability")
    plt.ylabel("Frequency")

    # Histograma para la clase negativa.
    plt.subplot(1, 2, 2)
    plt.hist(neg_probs, bins=n_bins, color='red', edgecolor='black')
    plt.title("Histogram of predicted probabilities\nfor negative class")
    plt.xlabel("Predicted probability")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    return {
        "array_passed_to_histogram_of_positive_class": pos_probs,
        "array_passed_to_histogram_of_negative_class": neg_probs,
    }



def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """
    y_true_binary = np.array([1 if label == positive_label else 0 for label in y_true])
    
    # Definir 11 umbrales igualmente espaciados entre 0 y 1.
    thresholds = np.linspace(0, 1, 11)
    
    # Listas para almacenar TPR y FPR para cada umbral.
    tpr_list = []
    fpr_list = []
    
    # Calcular TPR y FPR para cada umbral.
    for thresh in thresholds:
        # Clasificar como positivo cuando la probabilidad es mayor o igual al umbral.
        y_pred = (y_probs >= thresh).astype(int)
        
        # Calcular los componentes de la matriz de confusión.
        TP = np.sum((y_true_binary == 1) & (y_pred == 1))
        FP = np.sum((y_true_binary == 0) & (y_pred == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred == 0))
        TN = np.sum((y_true_binary == 0) & (y_pred == 0))
        
        # Calcular TPR (sensibilidad): TP / (TP + FN)
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # Calcular FPR: FP / (FP + TN)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Convertir las listas a arrays de NumPy
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    
    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_arr, tpr_arr, marker='o', linestyle='-', label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Retornar las tasas FPR y TPR
    return {"fpr": fpr_arr, "tpr": tpr_arr}
