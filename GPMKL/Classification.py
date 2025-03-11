import numpy as np
from scipy.optimize import minimize
from GPMKL.utils import Kernel  # Fixed import to use the correct module pathKernel



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
            #print(f"Step {i + 1}, NLL: {self.log_likelihood(mode, y)}")
        self.mode = mode
        return self

    def fit(self, X_train, y_train, bounds=None, n_steps: int = 30, n_iter: int = 50, num_init: int = 2):
        """Train the model using the provided data, optimizing kernel parameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            bounds: Parameter bounds for optimization
            n_steps: Number of steps for posterior computation
            n_iter: Number of iterations for each optimization attempt
            num_init: Number of different initializations to try
        """
        self.X_train = X_train
        self.y_train = y_train

        # Define the objective function for optimization
        def objective(params):
            if self.kernel_type == "rbf":
                self.kernel = Kernel.kernel(self.kernel_type, sigma_f=params[0], l=params[1])
            elif self.kernel_type == "linear":
                self.kernel = Kernel.kernel(self.kernel_type, c=params[0])
            try:
                self.posterior(X_train, y_train, n_steps)
                return -self.log_likelihood(self.mode, y_train)  # Negative log likelihood
            except np.linalg.LinAlgError:
                return np.inf  # Return infinity for invalid parameters

        # Set initial parameters based on kernel type
        if self.kernel_type == "rbf":
            initial_params = [
                self.kernel_params.get('sigma_f', 10.0),  # Start with larger initial values
                self.kernel_params.get('l', 5.0)
            ]
            if bounds is None:
                bounds = [(0.1, 50), (0.1, 50)]
        else:
            initial_params = [self.kernel_params.get('c', 0.1)]
            if bounds is None:
                bounds = [(0, 50)]

        best_result = None
        best_objective = np.inf
        
        for i in range(num_init):
            try:
                current_params = initial_params if i == 0 else [
                    np.random.uniform(low=b[0], high=b[1]) for b in bounds
                ]
                result = minimize(
                    objective, 
                    current_params, 
                    bounds=bounds, 
                    method='L-BFGS-B',
                    options={'maxiter': n_iter, 'ftol': 1e-5}
                )
                if result.fun < best_objective:
                    best_objective = result.fun
                    best_result = result
            except:
                continue

        if best_result is None:
            raise ValueError("Optimization failed for all initializations")

        optimized_params = best_result.x
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