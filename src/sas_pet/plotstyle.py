import matplotlib as mpl


def apply_plot_style(
    *,
    font_size: int = 10,
    tick_label_size: int = 8,
    figure_dpi: int = 150,
    savefig_dpi: int = 300,
):
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "axes.labelsize": tick_label_size,
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "figure.dpi": figure_dpi,
        "savefig.dpi": savefig_dpi,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.9,
        "grid.color": "white",
        "grid.linewidth": 1.0,
        "axes.facecolor": "whitesmoke",
    })


def spie_columnwidth_inch(columnwidth_pt: float = 242.5389) -> float:
    return float(columnwidth_pt) / 72.27