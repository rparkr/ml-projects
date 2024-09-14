# %% [markdown]
# # Sigmoid function demo
# This simple demonstration shows how the shape of the sigmoid
# function changes based on changes in the input parameters.

# %%
from IPython.display import display, clear_output
import ipywidgets as widgets
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(
    w: float = 0.0,
    x: float = 0.0,
    b: float = 0.0,
    num: float = 1.0,
    den: float = 1.0,
) -> np.array:
    """Compute the sigmoid function for an input array."""
    z = (w * x) + b
    return num / (den + np.exp(-z))


@widgets.interact(
    w=widgets.FloatSlider(min=-5, max=5, step=0.1, value=1.0, continuous_update=False),
    b=widgets.FloatSlider(min=-20, max=20, step=0.1, value=0.0, continuous_update=False),
    num=widgets.IntSlider(min=-10, max=10, step=1, value=1, continuous_update=False),
    den=widgets.IntSlider(min=-10, max=10, step=1, value=1, continuous_update=False),
    plot_backend=['matplotlib', 'hvplot'],
)
def plot_sigmoid(
    w: float = 1.0,
    b: float = 0.0,
    num: float = 1.0,
    den: float = 1.0,
    plot_backend='matplotlib',
) -> None:
    x = np.arange(start=-10, stop=10.1, step=1)
    sigma = sigmoid(w, x, b, num, den)
    df = pd.DataFrame(dict(x=x, y=sigma))
    if plot_backend=='hvplot':
        display(df.hvplot.line(x="x", y="y", title='Sigmoid function'))
        clear_output(wait=True)
    else:
        p = df.plot(x='x', y='y', title='Sigmoid function')
        display(plt.show(p))


# %%
