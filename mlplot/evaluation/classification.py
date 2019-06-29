"""Evaluation class for 2-class classification evaluation"""
import sklearn.metrics as metrics

from . import np, sp, plt
from .evaluation import ModelEvaluation
from ..errors import InvalidArgument

class ClassificationEvaluation(ModelEvaluation):
    """2-class classification model evaluation

    Parameters
    ----------
    y_true : list or 1d numpy array with elements 0 or 1
             A collection of the true labels
    y_pred : list or 1d numpy array with elements 0.0 to 1.0
             A collection of the prediction probabilities of class value 1.
    class_names : list with two string elements
                  These are the names of the two classes and they are used in plots.
                  The first string maps to the 0 class and the second to the 1 class.
    model_name : string
                 The name of the model is used when creating plots
    """

    def __init__(self, y_true, y_pred, class_names, model_name):
        super().__init__(y_true=y_true, y_pred=y_pred, model_name=model_name)

        # Check y_true values
        true_values = np.sort(np.unique(self.y_true))
        if len(true_values) != 2 or not np.equal(true_values, np.array([0, 1])).all():
            raise InvalidArgument('y_true must contain only values of 0 or 1')

        # Check y_pred values
        if ((self.y_pred < 0.0) | (self.y_pred > 1.0)).any():
            raise InvalidArgument('y_pred must contain only values between [0.0, 1.0]')

        # Check class names
        self.class_names = class_names
        if len(class_names) != 2 or not isinstance(class_names[0], str) or not isinstance(class_names[1], str):
            raise InvalidArgument('class_names must contain two class names (strings)')

    def __repr__(self):
        return 'ClassificationEvaluation(model_name={}, class_names=[{}, {}])'.format(self.model_name, self.class_names[0], self.class_names[1])

    def _binarize_pred(self, threshold=0.5):
        """Return the prediction as integers based on the threshold"""
        return (self.y_pred > threshold).astype(int)

    ###################################
    # Scores

    def roc_auc_score(self):
        """Return the ROC AUC score"""
        return metrics.roc_auc_score(y_true=self.y_true, y_score=self.y_pred)

    def average_precision_score(self):
        """Return the average precision score"""
        return metrics.average_precision_score(y_true=self.y_true, y_score=self.y_pred)

    def f1_score(self, threshold=0.5):
        """Return the f1 score

        Parameters
        ----------
        threshold : float within [0.0, 1.0]
                    Defines the cutoff to be considered in the asserted class
        """
        binarized = self._binarize_pred(threshold=threshold)
        return metrics.f1_score(y_true=self.y_true, y_pred=binarized)

    def accuracy_score(self, threshold=0.5):
        """Return the f1 score

        Parameters
        ----------
        threshold : float within [0.0, 1.0]
                    Defines the cutoff to be considered in the asserted class
        """
        binarized = self._binarize_pred(threshold=threshold)
        return metrics.accuracy_score(y_true=self.y_true, y_pred=binarized)

    ###################################
    # Plots

    def roc_curve(self, ax=None):
        """Plot a reciever operating curve

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        false_pos_rate, true_pos_rate, _ = metrics.roc_curve(y_true=self.y_true, y_score=self.y_pred)

        # Create figure
        label = '{0}|AUC={1:0.2}'.format(self.model_name, self.roc_auc_score())
        ax.plot(false_pos_rate, true_pos_rate, label=label)

        # Line for random
        ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed', label='random|AUC=0.5')

        # Styling
        ax.set_title('ROC Curve for {}'.format(self.model_name))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')

        return ax

    def calibration(self, n_bins='auto', ax=None):
        """Plot a calibration plot

        Calibration plots are used the determine how well the predicted values match the true value.

        This plot is similar to `sklean <http://scikit-learn.org/stable/modules/calibration.html>`_.

        Parameters
        ----------
        n_bins : int or string
                The number of bins to group y_pred. See `numpy.histogram <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html>`_
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        # Calculate bin counts
        counts, bin_edges = np.histogram(self.y_pred, bins=n_bins, range=(0, 1))
        bin_labels = np.digitize(self.y_pred, bin_edges[:-1]) - 1
        fraction_positive = np.bincount(bin_labels, weights=self.y_true) / counts
        centers = np.bincount(bin_labels, weights=self.y_pred) / counts

        # Used to measure how far off from fully calibrated https://en.wikipedia.org/wiki/Brier_score
        brier = metrics.brier_score_loss(y_true=self.y_true, y_prob=self.y_pred)

        # Plot fraction positive
        label = '{0}|Brier Score={1:0.3}'.format(self.model_name, brier)
        ax.plot(centers, fraction_positive - centers, label=label)

        # 95% Confidence interval
        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        z = sp.stats.norm.ppf(0.975)
        ci_range = z * np.sqrt(fraction_positive * (1 - fraction_positive) / counts)
        lower_range = fraction_positive - ci_range
        upper_range = fraction_positive + ci_range
        ax.fill_between(centers, lower_range - centers, upper_range - centers, alpha=0.25)

        # Plot 0 line
        ax.plot([0, 1], [0, 0], color='gray', linestyle='dashed', alpha=0.25, label='perfect|Brier Score=0.0')

        # Format plot
        ax.set_ylim(-0.5, 0.5)
        ax.set_title('Calibration for {}'.format(self.model_name))
        ax.set_ylabel('Bias (Faction of Positive - Binned Prediction)')
        ax.set_xlabel('Binned Predictions')
        ax.legend(loc='upper left')

        return ax

    def precision_recall(self, x_axis='recall', ax=None):
        """Plot the precision-recall curve

        An example of this plot can be found on `sklean <http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_.

        Parameters
        ----------
        x_axis : str 'recall' or 'threshold'
                Specify the x axis of the plot. Precision recall tends to come in 2 flavors, one precision vs recall and
                the other precision and recall vs threshold.
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        precision, recall, threshold = metrics.precision_recall_curve(y_true=self.y_true, probas_pred=self.y_pred)
        average_precision = self.average_precision_score()

        # Create the figure
        if x_axis == 'recall':
            label = '{0}|Avg Precision={1:0.2f}'.format(self.model_name, average_precision)
            ax.step(recall, precision, where='post', label=label)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')

        elif x_axis == 'threshold':
            label = '{0} recall|Avg Precision={1:0.2f}'.format(self.model_name, average_precision)
            ax.plot(threshold, recall[:-1], label=label)
            label = '{0} precision|Avg Precision={1:0.2f}'.format(self.model_name, average_precision)
            ax.plot(threshold, precision[:-1], label=label)
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Precision/Recall')

        else:
            raise InvalidArgument('x_axis can be either "recall" or "threshold"')

        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('Precision Recall for {}'.format(self.model_name))
        ax.legend()

        return ax

    def distribution(self, ax=None):
        """Plot histograms of the predictions grouped by class

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        alpha = 0.5  # Bars should be fairly transparent
        cond = self.y_true.astype(bool)
        label = '{}|class={}'.format(self.model_name, self.class_names[0])
        ax.hist(self.y_pred[cond], alpha=alpha, label=label)
        label = '{}|class={}'.format(self.model_name, self.class_names[1])
        ax.hist(self.y_pred[~cond], alpha=alpha, label=label)

        # Keep consistent x axis
        ax.set_xlim(0, 1)

        ax.legend()
        ax.set_title('Distribution of Predictions for {}'.format(self.model_name))
        ax.set_ylabel('Occurrences')
        ax.set_xlabel('Prediction')

        return ax

    def confusion_matrix(self, threshold=0.5, ax=None):
        """Plot a heatmap for the confusion matrix

        An example of this heatmap can be found on `sklean <http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py>`_.

        Parameters
        ----------
        threshold : float within [0.0, 1.0]
                    Defines the cutoff to be considered in the asserted class
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        binarized = self._binarize_pred(threshold=threshold)
        mat = metrics.confusion_matrix(self.y_true, binarized)

        # Heatmap
        ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)

        # Class names
        tick_marks = np.arange(len(self.class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.class_names)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.class_names)

        # Numbers in squares
        color_thresh = mat.max() / 2
        total = mat[:].sum()
        for (x, y), val in np.ndenumerate(mat):
            text_color = 'white' if val > color_thresh else 'black'
            text = '{0}\n({1:.1f}%)'.format(val, 100*val/total)
            ax.text(x, y, text, fontsize='large', horizontalalignment='center', color=text_color)

        ax.set_ylabel('Actual Class')
        ax.set_xlabel('Predicted Class | Threshold={}'.format(threshold))
        ax.set_title('Confusion Matrix for {}'.format(self.model_name))

        return ax

    def report_table(self, ax=None):
        """Generate a report table containing key stats about the dataset

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        tbl = []

        # Sample counts
        tbl.extend([
            ('Total observations', len(self.y_true)),
            ('Class "{}" count'.format(self.class_names[0]), int((self.y_true == 0).sum())),
            ('Class "{}" count'.format(self.class_names[1]), int((self.y_true == 1).sum())),
        ])

        # Scoring
        tbl.extend([
            ('F1 Score', self.f1_score()),
            ('Accuracy', self.accuracy_score()),
            ('AUC', self.roc_auc_score()),
            ('Avg Precision', self.average_precision_score()),
        ])

        self._format_table(table=tbl, ax=ax)
        ax.set_title('Classification Report for {}'.format(self.model_name))

        return ax