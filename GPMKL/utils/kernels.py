import numpy as np

class Kernel:
    # Dictionary to store kernel hyperparameters
    hyperparameters = {}

    @staticmethod
    def get_param(param, default):
        """
        Retrieve a parameter value from the hyperparameters dictionary.
        Args:
            param (str): The name of the parameter.
            default: The default value to return if the parameter is not found.
        Returns:
            The value of the parameter if found, otherwise the default value.
        """
        return Kernel.hyperparameters.get(param, default)

    @staticmethod
    def linear(x1, x2, c=1.0):
        """
        Linear kernel function with a multiplicative constant term.
        Args:
            x1 (np.ndarray): An array of m points (m x d).
            x2 (np.ndarray): An array of n points (n x d).
            c (float): Multiplicative constant.
        Returns:
            np.ndarray: An (m x n) matrix representing the scaled dot product of x1 and x2.
        """
        return c * (x1 @ x2.T)

    @staticmethod
    def rbf(x1, x2, sigma_f=1.0, l=1.0):
        """
        Radial Basis Function (RBF) kernel, also known as the Gaussian kernel.
        Args:
            x1 (np.ndarray): An array of m points (m x d).
            x2 (np.ndarray): An array of n points (n x d).
            sigma_f (float): Kernel vertical variation parameter.
            l (float): Kernel length parameter.
        Returns:
            np.ndarray: An (m x n) matrix representing the kernel matrix between x1 and x2.
        """
        sq_dist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sq_dist)

    @classmethod
    def kernel(cls, kernel_type, **kwargs):
        """
        Method to select and configure a kernel function based on the kernel type.
        Args:
            kernel_type (str): The type of kernel to use ('linear' or 'rbf').
            **kwargs: Arbitrary keyword arguments for kernel hyperparameters.
        Returns:
            A callable representing the configured kernel function.
        Raises:
            ValueError: If an invalid kernel type is provided.
        """
        cls.hyperparameters = kwargs  # Update hyperparameters
        if kernel_type == "rbf":
            return lambda x1, x2: cls.rbf(x1, x2, **cls.hyperparameters)
        elif kernel_type == "linear":
            return lambda x1, x2: cls.linear(x1, x2, **cls.hyperparameters)
        else:
            raise ValueError("Invalid kernel type")