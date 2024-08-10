import numpy as np
import scipy.stats as stats
from scipy.special import comb


def graph_binomial_distribution(N, p, k):
    """
    Yields probability of k successful cases in N total trials
    for a p probability of success.
    """
    N = np.floor(N)
    # We shift k so that the steps of the function are centered at floor(k).
    k = np.floor(k + 0.5)
    return comb(N, k) * p ** k * (1 - p) ** (N - k)


def graph_poisson_distribution(x, lamb):
    # We shift x to line up with integers.
    x = np.floor(x + 0.5)
    return stats.poisson.pmf(x, lamb)


def hist_to_pmf(ax, X, **kwargs):
    """
    Plots a normalized histogram so that the height of the
    rectangles equals the probability of falling within
    that rectangle.
    """
    occurrences, limits, rectangles = ax.hist(X, **kwargs)

    total_occurrences = sum(occurrences)
    for n, r in zip(occurrences, rectangles):
        r.set_height(n / total_occurrences)
    new_rectangle_heights = [r.get_height() for r in rectangles]
    ax.set_ylim(0, 1.15 * np.max(new_rectangle_heights))


def hist_to_pdf(ax, X, **kwargs):
    """
    Draws a histogram where the area of each rectangle
    is the probability of falling within.
    axis: ax
    X: data
    """
    freqs, bins, rects = ax.hist(X, **kwargs)
    total = sum(freqs)
    for r, freq in zip(rects, freqs):
        # As we're approximating a pdf, the height of the rectangle times
        # its width must approximate the probability of falling within
        # that rectangle. That is, height * width = freq / total.
        r.set_height((freq / total) / r.get_width())
    max_height = np.max([(freq / total) / r.get_width()
                         for r, freq in zip(rects, freqs)])
    ax.set_ylim(0, 1.25 * max_height)
