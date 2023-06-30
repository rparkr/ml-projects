'''Train a KMeans classifier on any numeric dataset, visualize the output
and training progress, make predictions for new data, and generate color
palettes from images.

This is a vectorized implementation of the k-means clustering algorithm with
a convenient and robust API to classify data from any NumPy dataset and to
generate color palettes from images.

K-Means is an algorithm that finds groups, or _clusters_, in a dataset by:
1. choosing $k$ starting data points as the cluster centers
2. assigning all remaining data points to a cluster based on the cluster
   center nearest to them
3. computing the new cluster center as the mean of all the points in
   that cluster
4. repeating steps 2-3 until convergence, which occurs when no points are
   reassigned after successive iterations

**What is it useful for?**  
Clustering algorithms are _unsupervised_: there is no label, or target, to
predict. Instead, the algorithm seeks to find a pattern in the data, which can
help group data into simlar categories. Some example uses of clustering include:
- find related items (products, articles, observations) to an item of interest
- group text documents, like comments or reviews, into categories (topics)
- segment customers
- compress images by reducing the colors used
- categorize transactions

This module is useful for teaching about the KMeans algorithm by observing the
training process: just call `kmeans.plot_animate(data)` on a trained instance
of the `KMeans` class to watch the algorithm iteratively classify samples based
on proximity to clusters, then update the clusters to the mean of assigned
samples until converging.

Like the other projects in my _ML from scratch_ series, I implemented the
algorithm without referencing anyone else's code, as an exercise for me to
deeply understand common ML techniques.

Created by [Ryan Parker](https://github.com/rparkr)), June 2023.'''

# Built-in modules
from pathlib import Path  # create a folder for saving plots of each training iteration
import sys  # check whether the code is run in a Jupyter notebook (for the animated plot function)
import time  # pause code execution, for creating plot animations
import urllib.request  # download images using urllib.request.urlopen()

# Optional: ignore the "mean of empty slice" RuntimeWarning issued by NumPy
# when a cluster has no points assigned to it, when calculating average
# distance in the KMeans.fit() method. This usually happens when creating
# clusters from an image that has large sections of the same color, such as a
# pure black or white background. A partial workaround is to set
# `unique_start = True` when calling the `kmeans.fit()` method.
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# External packages
from IPython.display import clear_output  # create simple animations with multiple Matplotlib plots
import matplotlib.colors as mcolors  # for accessing color names and choosing color palettes
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # for drawing boxes on plots
import numpy as np
from PIL import Image  # the pillow package for loading images into np arrays
from tqdm.auto import tqdm  # display progress bars



class KMeans():
    '''Create KMeans estimators and plot results after training on a dataset.'''
    def __init__(
            self,
            k: int = 4,
            n_rounds: int = 10,
            max_iter: int = 100,
            threshold: float = 0.001,
            cluster_axis: int = -1,
            save_hist: bool = False,
            verbose: bool = False,
            random_state = None) -> None:
        '''Create an estimator to perform K-Means clustering on a dataset.

        To determine clusters, the K-Means algorithm:
        1. chooses `k` starting data points as the cluster centers
        2. classifies each data point by the cluster center nearest to it
        3. updates the cluster centers to the mean of all the points in that
           cluster
        4. repeats steps 2-3 until convergence: when cluster centers move less
           than `threshold` or when `max_iter` iterations are reached.
        
        # Example
        ```python
        >>> import numpy as np
        >>> import kmeans as km
        >>> rng = np.random.default_rng(seed=0)
        >>> X = np.concatenate((
        ...     rng.integers(low=20, high=30, size=(5, 2)) / 10,
        ...     rng.integers(low=0, high=10, size=(5, 2)) / 10))
        >>> kmeans = km.KMeans(k=2, random_state=0)
        >>> kmeans.fit(X)
        >>> kmeans.centers_
        array([[2.34 , 2.32 ],
               [0.7  , 0.775]])
        >>> kmeans.predict(np.array([[0.2, 0.4], [1.9, 2.3]]))
        array([1, 0], dtype=int64)
        
        ```

        # Parameters
        `k`: int, default=4
            The number of clusters to create.
        `n_rounds`: int, default=10
            The number of rounds for the clustering algorithm to run, each
            starting with randomly-chosen initial centers. The best-performing
            run will be returned, as determined by the set of clusters that
            minimizes the average squared distance between points and cluster
            centers.
        `max_iter`: int, default=100
            The maximum number of iterations in a single round. If convergence
            isn't reached by this number of iterations, the algorithm will move
            on to the next round.
        `threshold`: float, default=0.001
            Set the stopping criteria for model fit: if the average change
            for all cluster centers from the previous iteration to the current
            iteration is below this threshold, the algorithm is converged
            and no futher iterations will be peformed in the current round.
            Note that the dataset is normalized on a scale of 0-1 prior to
            running the algorithm, so this convergence criteria applies to data
            on that scale. The default threshold of 0.001 is a 0.1% change in
            cluster centers on each dimension.
        `cluster_axis`: int, default=-1
            The axis to use for clustering; all other axes will be treated as
            samples in the row dimension. The KMeans algorithm uses
            a two-dimensional dataset: each row is a sample, and the columns
            are the features for that sample. If the given dataset has
            more than two axes, the data will be reshaped into 2-D prior to
            fitting the KMeans algorithm. The cluster_axis parameter sets the
            axis that will be used as the columns axis for clustering; all other
            axes will be reshaped into the row dimension. For example, an image
            has 3 axes: height, width, RGB. By default, KMeans will reshape
            the height and width axes into the row dimension and will use the
            RGB values as columns, thus creating a reshaped array of shape:
            (height x width, 3) where the clusters are pixels in RGB space.
        `save_hist`: bool, default=False
            Whether to store data on the cluster centroids during each training
            iteration from model fit. Use only when you plan to use the
            `plot_animate()` method to show training progress, since the history
            increases the memory used by the KMeans object.
        `verbose`: bool, default=False
            If `True`, print outputs showing the progress of model fit.
        `random_state`: int, default=None
            If set, this value will be used as the seed when initializing
            cluster centers.
        
        # Attributes
        `best_round_`: the clustering round that had the smallest avg distance
            between points and their assigned cluster centers.
        `centers_`: the centers of the `k` clusters based on the best
            fit learned after performing `n_rounds` of the algorithm.
        `distances_`: the average distance (array norm) from each cluster
            center to the points assigned to that cluster.
        `data_min_`: the minimum value of the training data passed to .fit().
            Used for normalizing data before running the algorithm.
        `data_max_`: the maximum value of the training data passed to .fit().
            Used for normalizing data before running the algorithm.
        `hist_`: a 3-d array of the cluster centers from each training
            iteration of the best clustering round. Only available if
            `save_hist=True`. Used by the `plot_animate()` method to create
            an animation of training progress.
            Shape: (num_iter, num_clusters, num_features)
        `num_iters_`: the number of iterations before convergence in the best
            clustering round.

        # Methods
        `fit(X)`: train the model to learn the best cluster centers based on
            a dataset (`X`, a NumPy array)
        `classify(X)`: assign cluster labels to data based on the cluster
            centers learned through `fit()`
        `predict(X)`: alias for `classify(X)`
        `plot_scatter(X)`: display a Matplotlib scatter plot of the clusters.
        `plot_animate(X)`: display an animated Matplotlib scatter plot showing
            training progress and final cluster assignments. Requires
            `save_hist` to be set to `True` prior to model fit.
        '''
        # Store arguments passed on instantiation
        self.__dict__.update(locals())


    def _preprocess(
            self,
            data: np.ndarray,
            features_selected: list | None = None,
            normalize: bool = True,
            set_min_max: bool = False) -> np.ndarray:
        '''Normalize data and ensure it is in the proper 2-D format.'''
        # Copy the data first to ensure the input data isn't modified
        data = data.copy()
        if data.ndim == 1:
            raise ValueError("Dataset must have ≥2 dimensions; provided dataset has 1 dimension.")
        if data.ndim > 2 or self.cluster_axis not in [-1, data.ndim - 1]:
            if self.verbose: print(f"Reshaping input data from: {data.shape} to ", end='')
            data = np.reshape(data, (-1, data.shape[self.cluster_axis]))
            if self.verbose: print(f"{data.shape}")
        
        if set_min_max:
            # During training (.fit()), store min and max values per feature
            self.data_min_ = np.min(data, axis=0)
            self.data_max_ = np.max(data, axis=0)

        # Normalize data on a scale of 0-1 (based on the original scale of training data seen during .fit())
        if normalize:
            if features_selected:
                data = (data - self.data_min_[features_selected]) / (self.data_max_[features_selected] - self.data_min_[features_selected])
            else:
                data = (data - self.data_min_) / (self.data_max_ - self.data_min_)

        return data


    def _unnormalize(
            self,
            data: np.ndarray,
            features_selected: list | None = None):
        '''Return data to its original scale.'''
        if features_selected:
            return (data * (self.data_max_[features_selected] - self.data_min_[features_selected])) + self.data_min_[features_selected]
        else:
            return (data * (self.data_max_ - self.data_min_)) + self.data_min_


    def fit(self, data: np.ndarray, unique_start: bool=False) -> None:
        '''Learn cluster centers based on a dataset.

        After fitting the model, assign labels to the dataset by running
        `classify(X)`, which returns a NumPy integer array with the label for
        each sample (row) in the provided dataset, `X`.
        
        # Parameters
        `data`: `numpy.ndarray`
            The input array, to be used for grouping into clusters.
        `unique_start`: bool, default=`False`
            If `True`, the starting clusters will be selected from unique
            elements in the `data` array. This can be helpful when there are
            many points that are the same, like an image with large regions of
            the same color (e.g., black or white background).
            The default behavior (`False`) selects starting cluster centers at
            random from among all points. If there are many points that are the
            same, the algorithm can get stuck if two starting clusters happen to
            have the same values.
            For large datasets (e.g., >100,000 samples), setting this to `True`
            will noticeably degrade performance because NumPy must first
            sort the data and then find unique elements.
        '''
        # Store max and min values of training data, used for normalizing
        # based on min and max values of each column
        data = self._preprocess(data, set_min_max=True)

        if self.verbose:
            pbar = tqdm(total=self.n_rounds)

        if unique_start:
            _, unique_idxs = np.unique(data, return_index=True, axis=0)

        # Capture the round with the lowest avg distance from points to cluster
        # centers, starting with round 0 (by setting "best" to infinity)
        best_avg_distance = np.inf

        for round in range(self.n_rounds):
            if self.random_state != None:
                # Increment the random state each iteration to change starting
                # point selection while maintaining deterministic behavior
                self.random_state += 1
            rng = np.random.default_rng(seed=self.random_state)

            # ===============================
            # Select starting cluster centers
            # ===============================
            # Sample one point for each of the cluster centers. Shape: (num_clusters, num_features)
            if unique_start:
                centers = rng.choice(data[unique_idxs, :], size=self.k, replace=False, shuffle=False)
            else:
                centers = rng.choice(data, size=self.k, replace=False, shuffle=False)
            # Store in a list so later iterations can compare to prior ones
            centers_list = [centers]

            i = 0
            converged = False
            while not converged:
                if self.verbose:
                    pbar.set_description(f"Round {round + 1}/{self.n_rounds}, iter {i}")

                # Label samples based on closest cluster
                labels, distances = self.classify(
                    data, cluster_centers=centers, preserve_input_shape=False,
                    return_distances=True, preprocess=False)
                
                # =================================
                # Determine the new cluster centers
                # =================================
                # Create a boolean one-hot-encoding of the closest cluster center for each point.
                # Returns a 2d array of shape: (num_samples, num_clusters)
                indices = np.eye(self.k, dtype='bool')[labels]

                # Index the data array to compute the new mean based on the cluster assignments
                # cluster_assignments is a 3d array: (num_samples, num_features, num_clusters), where dim 3 is
                # 0s where the cluster was not assigned and equal to the column values at the assigned cluster.
                cluster_assignments = (data[..., np.newaxis] * indices[:, np.newaxis, :])
                centers = cluster_assignments.mean(axis=0, where=(cluster_assignments != 0)).T
                
                # =========================================
                # Step 5: check for convergence
                # =========================================
                if (np.all(centers == centers_list[-1])
                    or np.all(np.sqrt(np.sum((centers - centers_list[-1]) ** 2, axis=1)) < self.threshold)):
                    # If all points have the same labels as the last iteration or
                    # if cluster centers are within a tolerance of the prior iteration,
                    # the algorithm has converged.
                    converged = True
                else:
                    # Store centroid coordinates in a list for comparison to later iterations when checking for convergence
                    centers_list.append(centers)

                i += 1
                if i >= self.max_iter:
                    if self.verbose:
                        print(f"Round {round + 1}, max iter ({self.max_iter:,.0f}) reached. Proceeding to next round.")
                    converged = True
            
            # Store the best-performing round; starting with round 0.
            # Calculate the average distance from each point to its cluster center
            total_dist = np.sum(distances, axis=1) ** 0.5  # square root of sum of squared distances. Shape: (num_samples, num_clusters)
            avg_dists = np.mean(total_dist, axis=0, where=indices)  # shape: (num_clusters,)
            # If a cluster has no points assigned to it the mean is undefined (NaN), hence np.nanmean
            avg_dist = np.nanmean(avg_dists)  # scalar value, not an array
            
            if avg_dist < best_avg_distance:
                best_avg_distance = avg_dist
                self.best_round_ = round + 1 # the round with the lowest avg distance from points to assigned cluster centers
                self.distances_ = avg_dists  # normalized average distance, where each feature dimension has equal weight
                self.centers_ = self._unnormalize(centers)
                self.num_iters_ = len(centers_list)  # iterations until convergence
                if self.save_hist:
                    self.hist_ = self._unnormalize(np.array(centers_list))
                
            if self.verbose:
                pbar.update()  # update progress bar at end of round

        if self.verbose:
            pbar.close()  # remove progress bar


    def classify(
            self,
            data: np.ndarray,
            features_selected: list | None=None,
            cluster_centers: np.ndarray | None=None,
            preserve_input_shape: bool=True,
            return_distances: bool=False,
            preprocess: bool=True) -> np.ndarray:
        '''Classify each sample (row) in a dataset by labeling the sample
        with the cluster whose center is nearest to it.

        Returns a NumPy array with the cluster assigned to each sample.

        # Parameters
        `data`: `numpy.ndarray`
            The data to be classified. Must be 2-dimensional or `preprocess`
            must be set to `True` (default).
        `features_selected`: list of int, default=None
            When classifying data using a subset of the features, set this
            parameter to a list of the indices of the features to use, based on
            the 2-D version of the data: that is, the number of features is
            `training_data.shape[cluster_axis]`. For a 2-D array that would be
            the number of columns. For an image (a 3-D array), the default
            setting of `cluster_axis` (axis -1) would leave 3 features, one for
            each color channel, so the `features_selected` list could have
            values 0, 1, and 2.
            This is most useful for visualizing data in two or three dimensions,
            and is used by the plotting functions in this class.
            The default value `None` means that all features will be used.
        `cluster_centers`: np.ndarray, default=None
            If provided, these values will be used when computing the distance
            from each sample data point to the cluster centers. If `None`
            (default), the centers learned during model training (self.centers_)
            will be used.
            This is a 2-D array of shape: (n_clusters, n_features).
        `preserve_input_shape`: bool, default=True
            If `True` (default), the returned array will match the shape of
            the input data if ndims > 2. If `False` or if ndims == 2, a 1-D
            array will be returned with the cluster label for each sample, where
            the number of samples is given by the row dimension in the input
            data; or, if the data has >2 dimensions, the number of samples is
            given by the product of the shape of all axes of the input data
            besides the `cluster_axis` that is set at object initialization.
        `return_distances`: bool, default=False
            Whether to return a tuple of (labels, distances) arrays. Used in the
            .fit() method (model training).
        `preprocess`: bool, default=True
            Whether to normalize the provided data based on the min and max
            values from the training data seen during model fit.
        '''
        if data.ndim == 1:
            raise ValueError("Dataset must have ≥2 dimensions; given dataset has 1 dimension.")
        if preserve_input_shape:
            cluster_axis = self.cluster_axis if self.cluster_axis != -1 else data.ndim - 1
            input_shape = tuple(data.shape[i] for i in range(data.ndim) if i != cluster_axis)
        
        if np.any(cluster_centers):
            centers = cluster_centers
        else:
            centers = self.centers_
        
        if preprocess:
            data = self._preprocess(data, features_selected=features_selected)
            centers = self._preprocess(centers)
        
        if centers.size == 0:
            raise AssertionError("You need to fit the model before you can perform classification. Run KMeans.fit(X) on a dataset first.")

        # ================================
        # Compute distance to each cluster
        # ================================
        # Add a cluster dimension to the data array to store the distance of
        # each cluster center at each of the data array's dimensions.
        # Add a row dimension to the transposed centers array to broadcast down
        # the rows of the data array.
        # distances is a 3d array: (num_samples, num_features, num_clusters)
        if features_selected:
            # Use only the selected features when computing distance to each sample
            distances = (data[..., np.newaxis] - centers[:, features_selected].T[np.newaxis, ...]) ** 2
        else:
            distances = (data[..., np.newaxis] - centers.T[np.newaxis, ...]) ** 2

        # =============================================
        # Assign labels based on nearest cluster center
        # =============================================
        # .sum(axis=1) finds the distance from each point to each cluster center, returns a 2d array
        # .argmin(axis=1) finds the closest cluster center, returns a 1d array
        labels = distances.sum(axis=1).argmin(axis=1)

        if preserve_input_shape:
            return_val =  labels.reshape(input_shape)
        else:
            return_val = labels
        
        if return_distances:
            return_val = (return_val, distances)
        
        return return_val
    

    # Create aliases for classify()
    __call__ = classify  # Enables calling the object like a function
    predict = classify


    # ==============================
    # Plotting methods
    # ==============================
    def plot_scatter(
            self,
            data: np.ndarray,
            features_selected: list | None = None,
            feature_names: list | None = None,
            iteration: int | None = None,
            colors: list | None = None,
            show_boundary_lines: bool = False,
            show_cluster_regions: bool = False,
            limit_features: bool = False,
            filepath: str | None = None,
            dpi: int = 200,
            return_colors: bool = False,
            iteration_for_centers: int | None = None,
            sample_size: int | None = None,
            random_state: int | None = None
            ) -> list | None:
        '''Display a scatter plot showing the assigned cluster for each sample.
        
        # Parameters
        `data`: np.ndarray
            The data to be plotted. The recommendation is to use the same data
            that the model was trained on.
        `features_selected`: list of int, default = `None`
            A list of indices of the two features to use for creating the
            scatter plot. If no features are provided, a plot will be created
            using the first two features of `data`, after reshaping if needed
            to a 2-dimensional dataset of shape: num_samples, num_features.
        `feature_names`: list of str, default = `None`
            The names of the two selected features, to be placed on the x and y
            axes, respectively. If not provided, the column number will be used
            instead (e.g., default would be x=col_0, y=col_1).
        `iteration`: int, default = `None`
            Set the iteration number used for the cluster centers. Defaults to
            the final iteration in the best clustering round during model
            training (self.num_iters_).
        `colors`: {list, 'image'}, default = `None`
            If provided, this is a list of colors in any format accepted by
            `matplotlib`'s plotting methods (e.g., RGB values, HTML color names,
            hex values). Must have the same number of elements as the number of
            clusters. If there are more colors in this list than the number of
            clusters, only the first `k` colors will be used. If there are fewer
            colors in this list than the number of clusters, an error is raised.
                If `colors` is set to the string 'image', the colors of the plot
            will correspond to the actual cluster colors at the given iteration.
            If using `image`, the input data must be an image in RGB color space 
        `show_boundary_lines`: bool, default = `False`
            Whether to draw boundary lines showing the region covered by each
            cluster in 2-dimensional space.
        `show_cluster_regions`: bool, default = `False`
            Whether to shade the areas covered by each cluster.
        `limit_features`: bool, default = `False`
            Whether to limit the features used for classifying each data point
            to the two features set in `features_selected`. See Notes below for
            more information.
        `filepath`: str, default = `None`
            If provided, this is the filepath where an image of the plot will
            be saved before displaying the image.
        `dpi`: int, default = 200
            The resolution of the displayed (and saved) plot, in dots-per-inch.
            150 or above is recommended for high-definition display.
        `return_colors`: bool, default = False
            Whether to return a list of the colors used, to create consistency
            for subsequent plot calls (e.g., creating an animated plot).
        `iteration_for_centers`: int, default = `None`
            If set, use the centers from this iteration when overlaying
            the centers on top of the scatter plot. This is used in the
            plot_animate() function to show how centers update based on the mean
            of assigned data points.
            If `None` (default) use the centers from `iteration`.
        `sample_size`: int, default = `None`
            If set to an integer value, `sample_size` data points will be
            drawn uniformly from the input data prior to plotting, which can
            speed up plot time if the input data is large (>20k samples).
        `random_state`: int, default = `None`
            If `sample_size` is provided, set `random_state` to an integer
            for reproducible results. If `None`, then `self.random_state` will
            be used. If `self.random_state` is `None`, then plot will have a
            different sample of points each time it is run.
            
        # Notes
        When showing boundary lines or cluster regions, the cluster assigned to
        each data point may differ from the cluster assignment when not showing
        boundary lines or cluster regions. This is because boundary lines and
        cluster regions are computed using only two features, so cluster
        assignments are also computed using only two features when either of
        those parameters are set to `True`. When both `show_boundary_lines` and
        `show_cluster_regions` are set to `False` (default), the cluster
        assignment will be based on all features in the dataset unless the
        `limit_features` parameter is set to `True`.
        '''
        fig, axs = plt.subplots(nrows=1, ncols=2, dpi=dpi,
                        figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
        
        if iteration != None:
            centers = self.hist_[iteration] if iteration < self.num_iters_ else self.hist_[-1]
            iteration = f"{iteration:,.0f}/{self.num_iters_ - 1:,.0f}"
        else:
            iteration = f"{self.num_iters_ - 1:,.0f}"
            centers = self.centers_

        # Select k random colors for plotting
        if colors == None:
            rng = np.random.default_rng()
            colors = rng.choice(list(mcolors.CSS4_COLORS.keys()), size=self.k, replace=False).tolist()
        elif colors == 'image':
            colors = (centers / 255).tolist()
        else:
            if len(colors) > self.k:
                colors = colors[:self.k]
            elif len(colors) < self.k:
                raise ValueError(f"Provided list of colors must have the same number of elements as the number of clusters ({self.k}), but only {len(colors)} colors were provided.")
        
        if features_selected:
            if len(features_selected) != 2:
                raise ValueError(f"When provided, `features_selected` must be a list of two integers representing the dataset features to use for the x and y axes in the plot. {(len(features_selected))} features were given.")
        else:
            features_selected = [0, 1]
        
        col_a, col_b = features_selected

        # Create a copy of the data in 2 dimensions
        data = self._preprocess(data, normalize=False)

        if sample_size and data.shape[0] > sample_size:
            rng = np.random.default_rng(seed=random_state if random_state != None else self.random_state)
            data = rng.choice(data, sample_size, replace=False)
        
        # =======================================
        # Plot boundary lines and cluster regions
        # =======================================
        if show_boundary_lines or show_cluster_regions:
            # Show boundary lines between clusters using a countour plot
            # See: https://matplotlib.org/stable/plot_types/arrays/contour.html
            # Add 10% to each boundary to provide some blank space around the furthest points
            x = np.linspace(np.min(data[:, col_a])*1.1, np.max(data[:, col_a])*1.1, 100)
            y = np.linspace(np.min(data[:, col_b])*1.1, np.max(data[:, col_b])*1.1, 100)
            xy_grid = np.meshgrid(x, y, indexing='xy')
            z = np.c_[xy_grid[0].reshape(-1), xy_grid[1].reshape(-1)]  # (num_samples, 2); i.e., (10_000, 2)

            labels = self.classify(data=z, features_selected=features_selected, cluster_centers=centers, preprocess=True)
            labels = labels.reshape((100, 100))
        
        if show_boundary_lines:
            axs[0].contour(x, y, labels, levels=np.arange(self.k), colors=colors, alpha=0.5)
        if show_cluster_regions:
            # Fill background to show cluster assignments; this entire loop occurs
            # before the scatterplot loop so the scatterplot points are drawn on top of
            # these background points.
            for i in range(self.k):
                row_mask = (labels.reshape(-1) == i)
                axs[0].scatter(z[row_mask, 0], z[row_mask, 1], s=50, color=colors[i], alpha=0.1, label=i)

        # ================================
        # Plot data and predicted clusters
        # ================================
        if show_boundary_lines or show_cluster_regions or limit_features:
            # Classify points based only on the provided dimensions, so cluster
            # assignments match the cluster boundaries
            dataset_labels = self.classify(data[:, features_selected], features_selected=features_selected, cluster_centers=centers, preprocess=True)
        else:
            dataset_labels = self.classify(data, cluster_centers=centers, preprocess=True)
        counts = np.eye(self.k, dtype='int')[dataset_labels].sum(axis=0)
        for i in range(self.k):
            # Plot the points and their categories
            row_mask = (dataset_labels == i)
            axs[0].scatter(data[row_mask, col_a], data[row_mask, col_b], s=50, color=colors[i], alpha=0.5, label=i)
            # Plot the distribution of labels
            axs[1].bar(x=i, height=counts[i], color=colors[i], edgecolor='k', linewidth=2, label=i)
            y_midpoint = np.max(counts) / 2
            x_midpoint = self.k / 2
            axs[1].add_patch(Rectangle((i - x_midpoint/10, y_midpoint/10), x_midpoint/5, y_midpoint/5, color='black'))
            axs[1].text(x=i, y=y_midpoint/5, s=i, fontweight='bold', fontsize='18',
                    ha='center', va='center', color='white')

        # Plot the center points on top of everything else
        if iteration_for_centers != None:
            # Use the centers from a different iteration, to show how they update based on the mean of assigned samples
            centers = self.hist_[iteration_for_centers]
        axs[0].scatter(centers[:, col_a], centers[:, col_b], color=colors, edgecolor='white',
                    s=200, linewidth=2, alpha=1.0, label="centers")
        
        axs[0].set_xlabel(feature_names[0] if feature_names else f"Feature {col_a}")
        axs[0].set_ylabel(feature_names[1] if feature_names else f"Feature {col_b}")
        axs[0].set_title('Cluster assignments and cluster centers')
        axs[1].set_title('Points per cluster')
        axs[1].axis('off')
        # Legend isn't necessary because the bar plot (axs[1]) is labeled
        # axs[1].legend(loc='upper right', bbox_to_anchor=(-0.02, 1.0), borderaxespad=0.0)
        fig.suptitle(f"Round: {self.best_round_}, Step: {iteration}")

        if filepath:
            # Save figure before display, since display clears the figure from memory
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        plt.show()

        # if self.verbose:
        #     colors_used = {i: colors[i] for i in range(len(colors))}
        #     print(f"Colors used: {colors_used}")

        if return_colors:
            return colors
    

    def plot_animate(
            self,
            data: np.ndarray,
            fps: float = 2.0,
            save_img_folder_path: str | None = None,
            features_selected: list | None = None,
            feature_names: list | None = None,
            colors: list | None = None,
            show_boundary_lines: bool = False,
            show_cluster_regions: bool = False,
            limit_features: bool = False,
            filepath: str | None = None,
            dpi: int = 200,
            sample_size: int = 5000,
            random_state: int | None = None
            ) -> None:
        '''Display an animated scatter plot showing the assigned cluster for
        each sample during each iteration of model fit.

        To use this method, `save_hist` must be set to `True` when instantiating
        the `KMeans` class.
        
        # Parameters
        `data`: np.ndarray
            The data to be plotted. The recommendation is to use the same data
            that the model was trained on.
        `fps`: float, default=2.0
            The number of frames to display per second, where each frame is
            one iteration in the best clustering round.
        `save_img_folder_path`: str, default=`None`
            The folder (relative to the current working directory) where images
            of each frame of the animated plot will be saved. If `None`
            (default), frames will not be saved when the plot is rendered.
        `features_selected`: list of int, default=`None`
            A list of indices of the two features to use for creating the
            scatter plot. If no features are provided, a plot will be created
            using the first two features of `data`, after reshaping if needed
            to a 2-dimensional dataset of shape: num_samples, num_features.
        `feature_names`: list of str, default = `None`
            The names of the two selected features, to be placed on the x and y
            axes, respectively. If not provided, the column number will be used
            instead (e.g., default would be x=col_0, y=col_1).
        `colors`: {list, 'image'}, default = `None`
            If provided, this is a list of colors in any format accepted by
            `matplotlib`'s plotting methods (e.g., RGB values, HTML color names,
            hex values). Must have the same number of elements as the number of
            clusters. If there are more colors in this list than the number of
            clusters, only the first `k` colors will be used. If there are fewer
            colors in this list than the number of clusters, an error is raised.
                If `colors` is set to the string 'image', the colors of the plot
            will correspond to the actual cluster colors at the given iteration.
            If using `image`, the input data must be an image in RGB color space 
        `show_boundary_lines`: bool, default=`False`
            Whether to draw boundary lines showing the region covered by each
            cluster in 2-dimensional space.
        `show_cluster_regions`: bool, default=`False`
            Whether to shade the areas covered by each cluster.
        `limit_features`: bool, default=`False`
            Whether to limit the features used for classifying each data point
            to the two features set in `features_selected`. See Notes below for
            more information.
        `filepath`: str, default = `None`
            If provided, this is the filepath where an image of the plot will
            be saved before displaying the image.
        `dpi`: int, default = 200
            The resolution of the displayed (and saved) plot, in dots-per-inch.
            150 or above is recommended for high-definition display.
        `sample_size`: int, default = 5000
            If set to an integer value, `sample_size` data points will be
            drawn uniformly from the input data prior to plotting, which can
            speed up plot time if the input data is large (>20k samples).
        `random_state`: int, default = `None`
            If `sample_size` is provided, set `random_state` to an integer
            for reproducible results. If `None`, then `self.random_state` will
            be used. If `self.random_state` is `None`, then a seed number will
            be chosen that will be used for all plots in the animation.
            
        # Notes
        When showing boundary lines or cluster regions, the cluster assigned to
        each data point may differ from the cluster assignment when not showing
        boundary lines or cluster regions. This is because boundary lines and
        cluster regions are computed using only two features, so cluster
        assignments are also computed using only two features when either of
        those parameters are set to `True`. When both `show_boundary_lines` and
        `show_cluster_regions` are set to `False` (default), the cluster
        assignment will be based on all features in the dataset unless the
        `limit_features` parameter is set to `True`.
        '''

        if not self.save_hist:
            raise AssertionError('Training history was not saved. Set `save_hist` to true and re-run `kmeans.fit()` to store training history.')
        
        # Check if this code is running in a notebook environment
        in_notebook = 'ipykernel' in sys.modules

        # Turn off verbose mode since the printed output will interrupt frame display
        verbosity = self.verbose
        self.verbose = False

        # Store color preference for images, if given
        if colors == 'image':
            use_image_colors = True
        else:
            use_image_colors = False

        if save_img_folder_path:
            num_digits = len(str(len(self.hist_)))  # label saved image files consistently using zero padding
            # Create a folder for storing the images
            folderpath = Path(save_img_folder_path)
            if not folderpath.exists():
                folderpath.mkdir(parents=True)
        
        # Convert the data to 2 dimensions
        data = self._preprocess(data, normalize=False)

        if sample_size and data.shape[0] > sample_size:
            if random_state != None:
                seed_val = random_state
            elif self.random_state != None:
                seed_val = self.random_state
            else:
                # Get a seed value
                rng = np.random.default_rng()
                seed_val = rng.integers(low=0, high=1000, size=1)[0]
            rng = np.random.default_rng(seed=seed_val)
            data = rng.choice(data, sample_size, replace=False)
        
        # Show the unlabeled starting points
        fig, axs = plt.subplots(nrows=1, ncols=2, dpi=dpi,
                        figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
        col_a, col_b = features_selected if features_selected else [0, 1]
        axs[0].scatter(data[:, col_a], data[:, col_b], s=50, color='darkgray', alpha=0.5)
        axs[0].set_xlabel(feature_names[0] if feature_names else f"Feature {col_a}")
        axs[0].set_ylabel(feature_names[1] if feature_names else f"Feature {col_b}")
        axs[0].set_title('Cluster assignments and cluster centers')
        axs[1].set_title('Points per cluster')
        axs[1].axis('off')
        fig.suptitle('Input data')
        if save_img_folder_path:
            filepath = folderpath / 'kmeans__input_data.png'
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.show()
        time.sleep(2)
        if in_notebook:
            clear_output(wait=True)
        else:
            plt.clf()  # clears the current figure, but doesn't remove it from memory. See: https://stackoverflow.com/a/33343289/17005348
            # plt.close(plt.get_fignums()[0])

        # Show each training iteration
        for i in range(len(self.hist_)):
            start_time = time.perf_counter()

            if save_img_folder_path:
                filepath = folderpath / f"kmeans_{str(i).zfill(num_digits)}a.png"
            
            colors = self.plot_scatter(
                data = data,
                features_selected = features_selected,
                feature_names = feature_names,
                iteration = i,
                colors = 'image' if use_image_colors else colors,
                show_boundary_lines = show_boundary_lines,
                show_cluster_regions = show_cluster_regions,
                limit_features = limit_features,
                filepath = filepath,
                dpi = dpi,
                return_colors = True
            )

            run_time = time.perf_counter() - start_time
            if run_time < (1 / fps):
                time.sleep((1 / fps) - run_time)
            if in_notebook:
                clear_output(wait=True)
            else:
                plt.clf()
            
            if i + 1 < self.num_iters_:
                # Show how centers update based on the mean of assigned points
                start_time = time.perf_counter()
                if save_img_folder_path:
                    filepath = folderpath / f"kmeans_{str(i).zfill(num_digits)}b.png"
                colors = self.plot_scatter(
                    data = data,
                    features_selected = features_selected,
                    feature_names = feature_names,
                    iteration = i,
                    colors = 'image' if use_image_colors else colors,
                    show_boundary_lines = show_boundary_lines,
                    show_cluster_regions = show_cluster_regions,
                    limit_features = limit_features,
                    filepath = filepath,
                    dpi = dpi,
                    return_colors = True,
                    iteration_for_centers = i + 1
                )

                run_time = time.perf_counter() - start_time
                if run_time < (1 / fps):
                    time.sleep((1 / fps) - run_time)
                if in_notebook:
                    clear_output(wait=True)
                else:
                    plt.clf()

        # Reset verbosity
        self.verbose = verbosity



# =====================================
# |     Color Palette Generation      |
# =====================================
def choose_clusters(
        data: np.ndarray,
        kmeans: KMeans,
        num: int = 5,
        weight_distinctness: float = 0.6,
        weight_size: float = 0.2,
        weight_cohesiveness: float = 0.2,
        wildcards: str | int | None = None) -> np.ndarray:
    '''Select a subset of the clusters from a trained KMeans classifier.

    Clusters are ranked by this formula:
    (size_weight * cluster_size_rank) + ((1 - size_weight) * cluster_cohesiveness_rank)
    
    where cluster_size_rank and cluster_cohesiveness_rank each have a max value
    of 1: for cluster_size that is the largest cluster (by number samples) and
    for cluster_cohesiveness that is the cluster with the smallest average
    distance to each of its points.

    The scores of all clusters are computed, then sorted in descending order,
    with the largest `num` clusters returned.
    
    # Parameters
    `data`: np.ndarray
        The data to be used for selecting the top clusters. This will likely be
        the data on which the `KMeans` classifier was trained. This is used to
        compute the distribution of poitns among clusters for ranking.
    `kmeans`: a trained instance of the `KMeans` class
        The trained KMeans classifier.
    `num`: int, default=5
        The number of clusters to return; a subset of the clusters trained by
        the `KMeans` classifer. A good rule of thumb is to set `num` to 1/3
        or 1/4 of the number of clusters trained by the `KMeans` classifier,
        for example, if the `KMeans` classifer has 20 clusters,
        `num` could be 5-7.
        If `None`, then the number of clusters returned will be 1/3 of the
        number of clusters trained by the `KMeans` classifier.
        If `num` is greater than or equal to the number of clusters trained by
        the classifier, then this function will return all clusters, ranked by
        the sorting function given above.
    `weight_distinctness`: float in the range [0, 1], default=0.6
        How much weight to place on the distinctness of a cluster: clusters
        that are most different from other clusters (e.g., distinct colors)
        receive better rankings.
    `weight_size`: float in the range [0, 1], default=0.2
        How much weight to place on the cluster sizes when ranking: larger
        clusters--those with more points--receive better rankings.
    `weight_cohesiveness`: float in the range [0, 1], default=0.2
        How much weight to place on cluster cohesiveness when ranking:
        clusters with lower average distance from the center to each point
        receive better rankings.
    `wildcards`: {int, 'all'}, default=`None`
        Chose `wildcard` number of clusters at random; the rest will be from
        the ranked order based on the scoring calculation above.
        If `'all'`, then all clusters will be sampled at random, ignoring
        the ranking order.
        If `None`, then no wildcards will be used.
    '''
    # Ensure that the number of clusters returned does not exceed the total number of clusters
    if num > len(kmeans.centers_):
        num = len(kmeans.centers_)
    if num == None:
        num = len(kmeans.centers_) // 3
    
    # Ensure weights add up to 1
    weights = np.array([weight_distinctness, weight_size, weight_cohesiveness])
    weight_distinctness, weight_size, weight_cohesiveness = weights / weights.sum()
    
    labels = kmeans.classify(data, preserve_input_shape=False)
    counts = np.bincount(labels)

    # Create the norm of each of the centers, for computing cosine similarity
    centers_norm = kmeans.centers_ / np.sqrt(np.sum(kmeans.centers_ ** 2, axis=1))[:, np.newaxis]
    # np.sqrt(np.sum(centers_norm ** 2, axis=1))  # norm should be 1.0 for each of the normalized clusters
    # Output: array([1., 1., 1., 1., ...])
    
    # Rank least-similar to most-similar clusters compared to every other cluster
    similarity_scores = np.argsort(np.matmul(centers_norm, centers_norm.T), axis=1)
    # Determine the most distinct clusters as the ones ranked least similar
    num_clusters = similarity_scores.shape[0]  # or, kmeans.k
    most_distinct = np.bincount(
        similarity_scores.flatten(order='F'),  # column-wise order
        weights=(  # the leftmost column, or least-similar, has a weight of 1; subsequent columns have reduced weights
            np.ones(similarity_scores.shape)
            * ((np.arange(num_clusters)[::-1] + 1) / num_clusters)).flatten(order='F'))
    
    scores = (
        (weight_distinctness * (most_distinct / most_distinct.max()))
        + (weight_size * (counts / counts.max()))
        + (weight_cohesiveness * (kmeans.distances_.min() / kmeans.distances_))
    )
    
    cluster_indices = np.argsort(scores)[::-1]  # sort in descending order

    # Select wildcards (random clusters)
    if wildcards != None:
        rng = np.random.default_rng()
        if wildcards == 'all':
            rng.shuffle(cluster_indices)  # shuffle in place
        else:
            # Choose 'wildcard' number of clusters at random and include with
            # the ranked clusters
            wildcards = np.clip(wildcards, a_min=0, a_max=len(counts))
            wildcard_idxs = rng.choice(cluster_indices[num - wildcards:], size=wildcards, shuffle=False, replace=False)
            cluster_indices = np.concatenate((cluster_indices[:num - wildcards], wildcard_idxs))
    
    return kmeans.centers_[cluster_indices[:num], :]


def find_nearest_named_colors(
        kmeans: KMeans,
        centers: np.ndarray | None = None,
        color_name_list: str='css4',
        color_space: str='rgb',
        comparison_color_space: str='rgb') -> dict:
    '''Return a dictionary with the nearest named colors and their RGB values
    
    # Parameters
    `kmeans`: instance of the KMeans class
        Must have already been trained on data using the .fit() method. Used
        for normalizing input data to the range [0, 1], which is used internally
        by Matplotlib.
    `centers`: np.ndarray
        The cluster centers. If not provided (default), `kmeans.centers_` will
        be used.
    `color_name_list`: {'css4', 'xkcd'}, default='css4'
        The color names (and associated hex values) to use. CSS4 colors are
        standard on the web, and XKCD colors come from the XKCD color survey.
        There are 148 CSS4 colors and 954 XKCD colors.
    `color_space`: {'rgb', 'hsv'}, default='rgb'
        The color space used when training the KMeans classifier.
    `comparison_color_space`: {'rgb', 'hsv'}, default='rgb'
        The color space to use when finding the nearest named color to each
        cluster center, irrespective of the color space the KMeans classifier
        was trained on.
        Using Eucliedean distance, HSV color space tends to be more accurate
        at finding perceptually similar colors than the RGB color space, but the
        differences are often small.
    '''
    # Use the cluster centers from the KMeans classifier if none are provided
    if not np.any(centers):
        centers = kmeans.centers_

    if color_name_list == 'xkcd':
        color_dict = mcolors.XKCD_COLORS
    else:
        color_dict = mcolors.CSS4_COLORS  # or, mcolors.cnames

    # Trim the alpha component using slicing to omit the 4th column: ([:, :3])
    rgb_named_colors = (mcolors.to_rgba_array(color_dict.values())[:, :3])  # Values are in the range [0, 1]
    
    # Use HSV colors to better match color perception. Inspired by: https://matplotlib.org/stable/gallery/color/named_colors.html#helper-function-for-plotting
    # The returned HSV values are normalized to the range [0, 1]
    hsv_named_colors = mcolors.rgb_to_hsv(mcolors.to_rgba_array(color_dict.values())[:, :3])
    
    if color_space == 'hsv':
        hsv_centers = kmeans._preprocess(centers, normalize=True)
        rgb_centers = mcolors.hsv_to_rgb(kmeans._preprocess(centers, normalize=True))
    else:
        hsv_centers = mcolors.rgb_to_hsv(centers / 255)
        rgb_centers = centers / 255
    
    if comparison_color_space == 'hsv':
        distances = (hsv_centers[..., np.newaxis] - hsv_named_colors.T[np.newaxis, ...]) ** 2
    else:
        distances = (rgb_centers[..., np.newaxis] - rgb_named_colors.T[np.newaxis, ...]) ** 2

    # .sum(axis=1) finds the distance from each cluster center to each named color, returns a 2d array: (num_clusters, num_named_colors)
    # .argmin(axis=1) finds the closest named, returns a 1d array: (num_clusters,)
    nearest_named_colors = distances.sum(axis=1).argmin(axis=1)

    return_dict = {}
    for i in range(len(nearest_named_colors)):
        color_idx = nearest_named_colors[i]
        return_dict[i] = {'name': list(color_dict.keys())[color_idx], 'rgb': (rgb_named_colors[color_idx, :] * 255).astype(np.uint8)}
    return return_dict


def download_img(url: str, reduced_img_max_size: int=128) -> np.ndarray:
    '''Download an image from a given URL and return it as a NumPy array.
    
    Returns a tuple of: (img_array, small_img_array) where `small_img_array` is
    a reduced-size version of `img_array` such that the longest dimension
    (height or width) of `img_array` is set to `reduced_img_max_size` pixels,
    with the shorter dimension adjusted proportionately.

    The `small_img_array` is used to speed up computation when fitting the
    KMeans classifier, while the `img_array` is used for displaying the image.
    '''
    # Reference: https://stackoverflow.com/questions/40911170/python-how-to-read-an-image-from-a-url/40911414#40911414
    with urllib.request.urlopen(url) as image_file:
        img = Image.open(image_file)
        # Resize the image so the longest side has 128 pixels, thus reducing computation time. See: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
        dims = np.array(img.size)  # width, height
        factor = np.max(dims / reduced_img_max_size)  # reduction factor
        
        if 'google.colab' in sys.modules:
            # Google Colab doesn't support the Image.Resample.LANCZOS property, so use Image.reduce() instead
            # Alternative method, using default resampling options. See: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.reduce
            # This sets the longest side to ~128 pixels, but not exactly.
            img_small = img.reduce(factor = int(factor))
        else:
            # Use the LANZCOS method for best resampling quality. See: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
            img_small = img.resize(tuple((dims // factor).astype(int)), resample=Image.Resampling.LANCZOS)
        
    return np.array(img), np.array(img_small)


def make_color_palette(
        url: str,
        kmeans: KMeans | None = None,
        num: int = 7,
        num_clusters: int | None = None,
        wildcards: str | int | None = None,
        weight_distinctness: float = 0.6,
        weight_size: float = 0.2,
        weight_cohesiveness: float = 0.2,
        save_img_folder_path: str | None = None,
        dpi: int = 150,
        fig_size: tuple=(12, 6),
        color_space: str = 'rgb',
        nearest_named_colors: bool = False,
        comparison_color_space: str ='rgb',
        color_name_list: str='css4',
        reduced_img_max_size: int=128,
        max_iter: int = 100,
        n_rounds: int = 5,
        save_hist: bool = True,
        verbose: bool = False,
        random_state: int | None = None) -> KMeans:
    '''Generate a color palette from an image using the KMeans algorithm.

    Returns a KMeans estimator fit to the input image along with the image array
    and a reduced-size version of the image.
    The returned tuple is: (kmeans, img_array, small_img_array)

    Example:
    ```python
    >>> import kmeans as km
    >>> kmeans, img_array, small_img_array = km.make_color_palette(
    ...   url = 'https://w.wiki/6t2f',
    ...   random_state = 0)
    >>> # Displays an image of the generated color palette
    >>> # Show the first two clusters
    >>> kmeans.centers_[:2].round()
    array([[148.,  83.,  58.],
           [191., 132.,  88.]])
    
    ```

    # Parameters
    `url`: str
        The URL to the image from which a color palette will be extracted.
    `kmeans`: instance of the `KMeans` class
        A fitted (trained) instance of the `KMeans` class. If provided, this
        function will skip training
    `num`: int, default=7
        The number of colors in the palette. A maximum of 12 colors can be
        plotted at one time.
        To visualize other selections of 12 colors at a time, set `num=12`
        and `wilcards='all'` and run the function multiple times, passing
        `kmeans=kmeans` on the second time and afterwards to avoid retraining
        the KMeans estimator each time. Each time the function is run, a new
        set of colors will be chosen.
    `num_clusters`: int, default=`None`
        The number of clusters (colors) to fit to the input image. As a general
        rule, this should be 3-4x the number of colors in the palette (`num`).
        If `None` (default), then 3x `num` clusters will be fit. Using more
        clusters can enhance the quality of generated palettes at the expense
        of slower computation. Computational speed can be improved by reducing
        `n_rounds`, `max_iter`, or `reduced_image_max_size`, although quality
        may also be reduced, especially if `reduced_image_max_size` is decreased
    `wildcards`: {int, 'all'}, default=`None`
        Chose `wildcard` number of clusters (colors) at random; the rest will
        be from a ranked ordering of the clusters from the KMeans classifier.
        If `'all'`, then all clusters will be sampled at random, ignoring
        the ranking order.
        If `None`, then no wildcards will be used.
    `weight_distinctness`: float in the range [0, 1], default=0.6
        How much weight to place on the distinctness of a cluster: clusters
        that are most different from other clusters (e.g., distinct colors)
        receive better rankings.
    `weight_size`: float in the range [0, 1], default=0.2
        How much weight to place on the cluster sizes when ranking: larger
        clusters--those with more points--receive better rankings.
    `weight_cohesiveness`: float in the range [0, 1], default=0.2
        How much weight to place on cluster cohesiveness when ranking:
        clusters with lower average distance from the center to each point
        receive better rankings.
    `save_img_folder_path`: str, default=`None`
        The folder (relative to the current working directory) where an
        image of the color palette will be saved before being displayed.
        If `None` (default), the rendered plot will not be saved.
    `dpi`: int, default = 150
        The resolution of the displayed (and saved) plot, in dots-per-inch.
        150 or above is recommended for high-definition displays.
    `fig_size: tuple of (width, height) int, default=(12, 6)
        The dimensions in inches for the generated plot.
    `color_space`: {'rgb', 'hsv'}, default='rgb'
        The color space to use when training the KMeans classifier. In some
        cases, HSV color space may perform better than RGB color space. For
        example, in images with large numbers of the same pixel value (like
        a pure black or white background), HSV color space may converge
        faster.
        Usually, the results are very similar in either color space.
    `nearest_named_colors`: bool, default=False
        Whether to plot the nearest named colors below the color palette.
    `comparison_color_space`: {'rgb', 'hsv'}, default='rgb'
        The color space to use when finding the nearest named color to each
        cluster center, irrespective of the color space the KMeans classifier
        was trained on.
        Using Eucliedean distance, HSV color space tends to be more accurate
        at finding perceptually similar colors than the RGB color space, but the
        differences are often small.
    `color_name_list`: {'css4', 'xkcd'}, default='css4'
        The list of named colors to use when finding nearest named colors.
        CSS4 colors are standard on the web, and XKCD colors come from the
        XKCD color survey.
        There are 148 CSS4 colors and 954 XKCD colors.
    `reduced_image_max_size`: int, default=128
        The maximum size in pixels for the longest dimension of the input
        image after reducing its size to speed up computation. The original
        image is maintained for plotting, while the reduced image is used
        to fit the KMeans classifier.
    `max_iter`: int, default=100
        The maximum number of iterations in a single round. If convergence
        isn't reached by this number of iterations, the algorithm will move
        on to the next round.
    `n_rounds`: int, default=5
        The number of rounds for the clustering algorithm to run, each
        starting with randomly-chosen initial centers. The best-performing
        run will be returned, as determined by the set of clusters that
        minimizes the average squared distance between points and cluster
        centers.
    `save_hist`: bool, default=`True`
        Whether to store a record of training progress in the fitted KMeans
        estimator. When set to `True`, the `kmeans.plot_animate()` method can
        be called to visualize progress in each iteration of model training.
        Increases memory usage, since the property `kmeans.hist_` will be
        populated with an array of cluster centers at each training iteration.
    `verbose`: bool, default=`False`
        If `True`, print output showing model training progress along with
        other information. Set to `False` (default) for fastest performance.
    `random_state`: int, default=None
        If set, this value will be used as the seed when initializing
        cluster centers.
    '''


    # Download image and convert to NumPy array
    img_array, small_img_array = download_img(url, reduced_img_max_size=reduced_img_max_size)

    # Train a new KMeans classifier if none is provided or if the number of
    # clusters it was trained with is fewer than the number of colors requested
    # for the color palette.
    if kmeans == None or kmeans.k < num:
        # Ensure the number of clusters is at least as many as the color palette
        if num_clusters == None:
            num_clusters = 3 * num
        else:
            num_clusters = max(num, num_clusters)

        # Instantiate the KMeans classifier
        kmeans = KMeans(
            k = num_clusters,
            max_iter = max_iter,
            n_rounds = n_rounds,
            save_hist = save_hist,
            verbose = verbose,
            random_state = random_state)

        # Train the KMeans classifier
        if verbose:
            print(f"Training KMeans classifier for {kmeans.n_rounds} rounds...")
        if color_space == 'hsv':
            # Omit the alpha channel (RGBA -> RGB) and place values in the range: [0, 1]
            small_hsv_array = mcolors.rgb_to_hsv(small_img_array[:, :, :3] / 255) 
            kmeans.fit(small_hsv_array, unique_start=True)
        else:
            kmeans.fit(small_img_array, unique_start=True)
        if verbose:
            print(f"Training complete. Converged in {kmeans.num_iters_} steps on round {kmeans.best_round_}/{kmeans.n_rounds}.")

    # Create the plot: image on left, with color palette on right.
    # A maximum of 12 colors can be displayed, 3 rows by 4 columns.
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_size, dpi=dpi)

    if color_space == 'hsv':
        cluster_colors = choose_clusters(
            data = small_hsv_array,
            kmeans = kmeans,
            num = num,
            wildcards = wildcards,
            weight_distinctness = weight_distinctness,
            weight_size = weight_size,
            weight_cohesiveness = weight_cohesiveness)
        cluster_colors = mcolors.hsv_to_rgb(
            kmeans._preprocess(cluster_colors, normalize=True)) * 255
    else:
        cluster_colors = choose_clusters(
            data = small_img_array,
            kmeans = kmeans,
            num = num,
            wildcards = wildcards,
            weight_distinctness = weight_distinctness,
            weight_size = weight_size,
            weight_cohesiveness = weight_cohesiveness)
    
    nearest_colors = find_nearest_named_colors(
        kmeans,
        centers = cluster_colors,
        color_name_list = color_name_list,
        color_space = color_space,
        comparison_color_space = comparison_color_space)

    w, h = axs[1].get_window_extent().width, axs[1].get_window_extent().height
    w, h = 100, (h/w) * 100  # Set width to 100 and height to be proportional to maintain a square grid

    swatch_size = 20  # max of 4 colums in a 100-unit space, with padding between each
    padding = 4
    num_patches = len(cluster_colors)
    nrows = np.ceil(num_patches / 4)
    y_start = (h - 24)

    for n, color_dict in enumerate(nearest_colors.values()):
        # Determine the row number
        row_num = np.ceil((n + 1) / 4) 
        # Determine whether the swatch is in the last row
        if row_num == nrows:
            if num_patches % 4 == 0:
                x_start = 4
            else:
                x_start = 4 + ((4 - (num_patches % 4)) * 12)  # based on number of swatches in the row
        else:
            x_start = 4
        # Choose text color based on perceived luminance. Calculation from: https://matplotlib.org/stable/tutorials/colors/colors.html#comparison-between-x11-css4-and-xkcd-colors
        luminance = np.sum(np.array([0.299, 0.587, 0.114]) * (cluster_colors[n] / 255))
        font_color = 'black' if luminance > 0.5 else 'white'
        axs[1].add_patch(
            Rectangle(
                xy=(x_start + ((n % 4) * 24), y_start - ((row_num - 1) * 36)),
                width=20,
                height=20,
                # edgecolor=font_color,
                # linewidth=0.5,
                facecolor=tuple(cluster_colors[n] / 255)))
        axs[1].text(
            x=x_start + ((n % 4) * 24) + 10,
            y=y_start - ((row_num - 1) * 36) + 12,
            s=mcolors.rgb2hex(cluster_colors[n] / 255),
            color=font_color,
            fontdict={'fontsize': 12, 'fontweight': 'normal'},
            ha='center', va='center')
        axs[1].text(
            x=x_start + ((n % 4) * 24) + 10,
            y=y_start - ((row_num - 1) * 36) + 6,
            s=tuple(np.round(cluster_colors[n]).astype(np.uint8)),
            color=font_color,
            fontdict={'fontsize': 10, 'fontweight': 'normal'},
            ha='center', va='center')

        # Nearest named color
        if nearest_named_colors:
            # Choose text color based on perceived luminance. Calculation from: https://matplotlib.org/stable/tutorials/colors/colors.html#comparison-between-x11-css4-and-xkcd-colors
            luminance = np.sum(np.array([0.299, 0.587, 0.114]) * (cluster_colors[n] / 255))
            font_color = 'black' if luminance > 0.5 else 'white'
            axs[1].add_patch(
                Rectangle(
                    xy=(x_start + ((n % 4) * 24), y_start - ((row_num - 1) * 36) - 10),
                    width=20,
                    height=8,
                    # edgecolor=font_color,
                    # linewidth=0.5,
                    facecolor=tuple(color_dict['rgb'] / 255)))
            axs[1].text(
                x=x_start + ((n % 4) * 24) + 10,
                y=y_start - ((row_num - 1) * 36) - 4,
                s=color_dict['name'] if color_name_list == 'css4' else color_dict['name'][5:],  # trim the prefix 'xkcd:'
                color=font_color,
                fontdict={'fontsize': 10},
                ha='center', va='center')
            axs[1].text(
                x=x_start + ((n % 4) * 24) + 10,
                y=y_start - ((row_num - 1) * 36) - 8,
                s=tuple(color_dict['rgb']),
                color=font_color,
                fontdict={'fontsize': 8},
                ha='center', va='center')

    axs[1].set_axis_off()
    axs[1].set_xlim(0, w)
    axs[1].set_ylim(0, h)
    axs[1].set_xticks(ticks=np.arange(start=0, stop=w, step=10))
    axs[1].set_yticks(ticks=np.arange(start=0, stop=h + 1, step=10))
    axs[1].grid(visible=False)
    axs[1].set_title(
        'Color palette\n(with closest named color below)' if nearest_named_colors else 'Color palette',
        color='white')

    axs[0].imshow(img_array)
    axs[0].set_title('Input', color='white')
    axs[0].set_axis_off()

    fig.set_facecolor('black')
    fig.suptitle('KMeans: Image → Color palette', 
                 color='white', fontdict={'size': 24, 'weight': 'light'})
    plt.tight_layout()

    if save_img_folder_path:
        # Create a folder for storing the image
        folderpath = Path(save_img_folder_path)
        if not folderpath.exists():
            folderpath.mkdir(parents=True)
        filepath = folderpath / f"kmeans_color_palette.png"
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')

    plt.show()

    return kmeans, img_array, small_img_array