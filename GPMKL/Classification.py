import numpy as np
from scipy.optimize import minimize
from .utils.kernels import Kernel

class GaussianProcessClassifier:
    def __init__(self, kernel_type='linear', noise=1e-10, **kernel_params):
        """Initialize the classifier with a specific kernel type and parameters."""
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.kernel = Kernel.kernel(kernel_type, **kernel_params)
        self.noise = noise  # Noise term to ensure the matrix is invertible
        self.trained = False

    def sigmoid(self, X):
        """Apply the sigmoid function to convert values to probabilities."""
        return 1 / (1 + np.exp(-X))

    def log_likelihood(self, X, y):
        """Compute the negative log likelihood for the latent function f."""
        return -np.sum(y * np.log(self.sigmoid(X)) + (1 - y) * np.log(1 - self.sigmoid(X)))

    def posterior(self, X, y, n_steps: int = 30):
        """ Find the mode of the posterior using Newton-Raphson method. """
        n = X.shape[0]
        mode = np.random.rand(n) * 0.1  # Small random initialization
        for i in range(n_steps):
            p = self.sigmoid(mode)
            W = p * (1 - p)
            H = np.diag(W)
            K = self.kernel(X, X) + self.noise * np.eye(len(X))  # Adding noise term for numerical stability
            mode += np.linalg.solve(H + np.linalg.inv(K), y - p - np.dot(H, mode))
            print(f"Step {i + 1}, NLL: {self.log_likelihood(mode, y)}")
        self.mode = mode
        return self

    def fit(self, X_train, y_train, bounds=None, n_steps: int = 30, n_iter: int = 50):
        """Train the model using the provided data, optimizing kernel parameters."""
        self.X_train = X_train
        self.y_train = y_train

        # Define the objective function for optimization
        def objective(params):
            if self.kernel_type == "rbf":
                self.kernel = Kernel.kernel(self.kernel_type, sigma_f=params[0], l=params[1])
            elif self.kernel_type == "linear":
                self.kernel = Kernel.kernel(self.kernel_type, c=params[0])
            self.posterior(X_train, y_train, n_steps)
            return -self.log_likelihood(self.mode, y_train)  # Negative log likelihood

        # Set initial parameters based on kernel type
        initial_params = [self.kernel_params.get('sigma_f', 1.0), self.kernel_params.get('l', 1.0)] if self.kernel_type == "rbf" else [self.kernel_params.get('c', 0)]
        if bounds is None:
            bounds = [(0.1, 10), (0.1, 10)] if self.kernel_type == "rbf" else [(0, 10)]

        # Perform optimization
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter': n_iter})
        optimized_params = result.x
        if self.kernel_type == "rbf":
            self.kernel_params = {'sigma_f': optimized_params[0], 'l': optimized_params[1]}
        elif self.kernel_type == "linear":
            self.kernel_params = {'c': optimized_params[0]}
        print("Optimized parameters:", self.kernel_params)

        self.trained = True
        return self

    def predict(self, X_test):
        """Predict using the trained model."""
        if not self.trained:
            raise ValueError("Fit the model first.")
        mean, _ = self.predictive_distribution(X_test)
        return (mean > 0).astype(int)

    def predict_proba(self, X_test):
        """Predict probabilities using the trained model."""
        if not self.trained:
            raise ValueError("Fit the model first.")
        mean, var = self.predictive_distribution(X_test)
        return self.sigmoid(mean), var

    def predictive_distribution(self, X_test):
        """ Compute the predictive distribution for new data points. """
        K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        k_star = self.kernel(self.X_train, X_test)
        p = self.sigmoid(self.mode)
        W = p * (1 - p)
        K_inv = np.linalg.inv(K + np.linalg.inv(np.diag(W)))
        mean = k_star.T @ K_inv @ (self.y_train - p)
        var = self.kernel(X_test, X_test) + self.noise * np.eye(len(X_test)) - k_star.T @ K_inv @ k_star
        return mean, np.diag(var)

    def classification_accuracy(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = (probabilities > 0.5).astype(int)
        return np.mean(predictions == labels)