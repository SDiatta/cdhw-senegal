"""
Visualization Module
Functions for visualizing CDHW analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Optional, Tuple


class CDHWVisualizer:
    """Visualize CDHW analysis results."""

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize visualizer.

        Parameters
        ----------
        figsize : tuple
            Default figure size (width, height).
        """
        self.figsize = figsize

    def plot_spatial_field(
        self,
        data: xr.DataArray,
        title: str,
        cmap: str = "RdBu_r",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        add_senegal_boundary: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot spatial field with map projection.

        Parameters
        ----------
        data : xr.DataArray
            2D spatial data (lat, lon).
        title : str
            Plot title.
        cmap : str
            Colormap name.
        vmin : float, optional
            Minimum value for colorbar.
        vmax : float, optional
            Maximum value for colorbar.
        add_senegal_boundary : bool
            Whether to add Senegal boundary.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.

        Returns
        -------
        tuple
            (fig, ax) - matplotlib figure and axes.
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = plt.axes(projection=ccrs.PlateCarree())
        else:
            fig = ax.get_figure()

        # Plot data
        im = data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={"label": data.attrs.get("long_name", "")}
        )

        # Add map features
        ax.coastlines(linewidth=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)

        if add_senegal_boundary:
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
            ax.set_extent([-17.5, -11.3, 12.3, 16.0], crs=ccrs.PlateCarree())  # Senegal bounds

        ax.set_title(title, fontsize=14, fontweight="bold")

        return fig, ax

    def plot_compound_events(
        self,
        compound_index: xr.DataArray,
        title: str = "Compound Drought-Heatwave Events",
        add_senegal_boundary: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot compound drought-heatwave events with discrete colors.

        Parameters
        ----------
        compound_index : xr.DataArray
            Compound event index (0-3).
        title : str
            Plot title.
        add_senegal_boundary : bool
            Whether to add Senegal boundary.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.

        Returns
        -------
        tuple
            (fig, ax) - matplotlib figure and axes.
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = plt.axes(projection=ccrs.PlateCarree())
        else:
            fig = ax.get_figure()

        # Custom colormap for events
        colors = ["white", "lightblue", "orange", "red"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        # Plot
        im = ax.pcolormesh(
            compound_index.lon,
            compound_index.lat,
            compound_index.isel(time=0),
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm
        )

        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05)
        cbar.set_label("Event Type")
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["None", "Drought", "Heat", "Compound"])

        # Add map features
        ax.coastlines(linewidth=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)

        if add_senegal_boundary:
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
            ax.set_extent([-17.5, -11.3, 12.3, 16.0], crs=ccrs.PlateCarree())

        ax.set_title(title, fontsize=14, fontweight="bold")

        return fig, ax

    def plot_time_series(
        self,
        data: xr.DataArray,
        title: str,
        label: str,
        ax: Optional[plt.Axes] = None,
        color: str = "blue"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot time series.

        Parameters
        ----------
        data : xr.DataArray
            1D time series.
        title : str
            Plot title.
        label : str
            Y-axis label.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.
        color : str
            Line color.

        Returns
        -------
        tuple
            (fig, ax) - matplotlib figure and axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        else:
            fig = ax.get_figure()

        ax.plot(data.time.values, data.values, color=color, linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_multi_field(
        self,
        data_dict: dict,
        title: str,
        cmap: str = "RdBu_r",
        add_senegal_boundary: bool = True
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot multiple spatial fields in subplots.

        Parameters
        ----------
        data_dict : dict
            Dictionary of {name: xr.DataArray} pairs.
        title : str
            Overall title.
        cmap : str
            Colormap name.
        add_senegal_boundary : bool
            Whether to add Senegal boundary.

        Returns
        -------
        tuple
            (fig, axes) - matplotlib figure and axes array.
        """
        n_plots = len(data_dict)
        ncols = min(2, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig = plt.figure(figsize=(14, 5 * nrows))
        axes = []

        for idx, (name, data) in enumerate(data_dict.items()):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection=ccrs.PlateCarree())
            axes.append(ax)

            # Plot
            im = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                add_colorbar=True,
                cbar_kwargs={"label": data.attrs.get("long_name", name)}
            )

            # Add map features
            ax.coastlines(linewidth=0.5)
            ax.gridlines(draw_labels=True, alpha=0.3)

            if add_senegal_boundary:
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
                ax.set_extent([-17.5, -11.3, 12.3, 16.0], crs=ccrs.PlateCarree())

            ax.set_title(name, fontsize=12, fontweight="bold")

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout()

        return fig, np.array(axes)

    def plot_index_comparison(
        self,
        spi: xr.DataArray,
        sti: xr.DataArray,
        location: Tuple[float, float],
        title: str = "SPI vs STI Time Series"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot SPI and STI comparison at a location.

        Parameters
        ----------
        spi : xr.DataArray
            Standardized Precipitation Index.
        sti : xr.DataArray
            Standardized Temperature Index.
        location : tuple
            (lat, lon) of location.
        title : str
            Plot title.

        Returns
        -------
        tuple
            (fig, ax) - matplotlib figure and axes.
        """
        lat, lon = location

        # Extract data at location
        spi_ts = spi.sel(lat=lat, lon=lon, method="nearest")
        sti_ts = sti.sel(lat=lat, lon=lon, method="nearest")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(spi_ts.time.values, spi_ts.values, label="SPI", color="blue", linewidth=2)
        ax.plot(sti_ts.time.values, sti_ts.values, label="STI", color="red", linewidth=2)

        # Add threshold lines
        ax.axhline(y=-0.5, color="blue", linestyle="--", alpha=0.5, label="Drought threshold")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Heat threshold")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

        # Shade compound event regions
        compound = (spi_ts < -0.5) & (sti_ts > 0.5)
        ax.fill_between(spi_ts.time.values, -3, 3, where=compound, alpha=0.2, color="purple",
                         label="Compound event")

        ax.set_xlabel("Time")
        ax.set_ylabel("Standardized Index")
        ax.set_title(f"{title} at ({lat:.2f}°N, {lon:.2f}°E)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
        """
        Save figure to file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filepath : str
            Output file path.
        dpi : int
            Resolution in dots per inch.
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {filepath}")
