"""
Drought and Heat Indices Module
Functions for calculating drought and heat indices from climate data.
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


class DroughtIndices:
    """Calculate drought indices from precipitation and temperature data."""

    @staticmethod
    def standardized_precipitation_index(
        precip: xr.DataArray,
        window: int = 12,
        min_periods: int = 1
    ) -> xr.DataArray:
        """
        Calculate Standardized Precipitation Index (SPI).

        Parameters
        ----------
        precip : xr.DataArray
            Precipitation data (time, lat, lon).
        window : int
            Window size in months (default: 12).
        min_periods : int
            Minimum observations in window.

        Returns
        -------
        xr.DataArray
            SPI values for each location and time.
        """
        # Reshape for easier calculation
        time, lat, lon = precip.shape
        precip_flat = precip.values.reshape(time, -1)

        spi = np.zeros_like(precip_flat)

        for i in range(precip_flat.shape[1]):
            series = precip_flat[:, i]
            valid_idx = ~np.isnan(series)

            if valid_idx.sum() < min_periods:
                spi[:, i] = np.nan
                continue

            # Calculate rolling mean and std
            rolling_mean = pd.Series(series).rolling(window, min_periods=min_periods).mean()
            rolling_std = pd.Series(series).rolling(window, min_periods=min_periods).std()

            # Standardize
            spi[:, i] = (series - rolling_mean) / rolling_std

        spi = spi.reshape(precip.shape)

        return xr.DataArray(
            spi,
            coords=precip.coords,
            dims=precip.dims,
            attrs={"long_name": f"Standardized Precipitation Index (SPI-{window})"}
        )

    @staticmethod
    def precipitation_anomaly(
        precip: xr.DataArray,
        reference_period: Optional[Tuple[str, str]] = None
    ) -> xr.DataArray:
        """
        Calculate precipitation anomaly.

        Parameters
        ----------
        precip : xr.DataArray
            Precipitation data (time, lat, lon).
        reference_period : tuple, optional
            (start_date, end_date) for reference climatology.
            If None, uses all available data.

        Returns
        -------
        xr.DataArray
            Precipitation anomalies.
        """
        if reference_period:
            ref_data = precip.sel(time=slice(*reference_period))
            climatology = ref_data.mean(dim="time")
        else:
            climatology = precip.mean(dim="time")

        anomaly = precip - climatology

        return anomaly.assign_attrs({
            "long_name": "Precipitation Anomaly",
            "units": precip.attrs.get("units", "mm")
        })

    @staticmethod
    def soil_moisture_deficit(
        precip: xr.DataArray,
        temp: xr.DataArray,
        capacity: float = 150.0
    ) -> xr.DataArray:
        """
        Calculate soil moisture deficit using a simple water balance.

        Parameters
        ----------
        precip : xr.DataArray
            Precipitation data (mm).
        temp : xr.DataArray
            Temperature data (°C).
        capacity : float
            Soil moisture holding capacity (mm). Default: 150 mm.

        Returns
        -------
        xr.DataArray
            Soil moisture deficit (0-1, where 1 is dry).
        """
        # Simple PET estimation (Thornthwaite-like)
        pet = 0.013 * temp.mean(dim="time") * 15  # Simplified ET estimation

        # Water balance: change in storage = precip - pet
        water_balance = precip - pet

        # Cumulative water deficit
        deficit = -water_balance.cumsum(dim="time")
        deficit = deficit.clip(min=0)

        # Normalize to 0-1
        smd = deficit / capacity
        smd = smd.clip(min=0, max=1)

        return smd.assign_attrs({
            "long_name": "Soil Moisture Deficit",
            "units": "dimensionless (0-1, 1=dry)"
        })


class HeatIndices:
    """Calculate heat indices from temperature data."""

    @staticmethod
    def temperature_anomaly(
        temp: xr.DataArray,
        reference_period: Optional[Tuple[str, str]] = None
    ) -> xr.DataArray:
        """
        Calculate temperature anomaly.

        Parameters
        ----------
        temp : xr.DataArray
            Temperature data (time, lat, lon).
        reference_period : tuple, optional
            (start_date, end_date) for reference climatology.
            If None, uses all available data.

        Returns
        -------
        xr.DataArray
            Temperature anomalies.
        """
        if reference_period:
            ref_data = temp.sel(time=slice(*reference_period))
            climatology = ref_data.mean(dim="time")
        else:
            climatology = temp.mean(dim="time")

        anomaly = temp - climatology

        return anomaly.assign_attrs({
            "long_name": "Temperature Anomaly",
            "units": temp.attrs.get("units", "°C")
        })

    @staticmethod
    def standardized_temperature_index(
        temp: xr.DataArray,
        window: int = 12,
        min_periods: int = 1
    ) -> xr.DataArray:
        """
        Calculate Standardized Temperature Index (STI).

        Parameters
        ----------
        temp : xr.DataArray
            Temperature data (time, lat, lon).
        window : int
            Window size in months (default: 12).
        min_periods : int
            Minimum observations in window.

        Returns
        -------
        xr.DataArray
            STI values (standardized departures).
        """
        time, lat, lon = temp.shape
        temp_flat = temp.values.reshape(time, -1)

        sti = np.zeros_like(temp_flat)

        for i in range(temp_flat.shape[1]):
            series = temp_flat[:, i]
            valid_idx = ~np.isnan(series)

            if valid_idx.sum() < min_periods:
                sti[:, i] = np.nan
                continue

            # Calculate rolling mean and std
            rolling_mean = pd.Series(series).rolling(window, min_periods=min_periods).mean()
            rolling_std = pd.Series(series).rolling(window, min_periods=min_periods).std()

            # Standardize
            sti[:, i] = (series - rolling_mean) / rolling_std

        sti = sti.reshape(temp.shape)

        return xr.DataArray(
            sti,
            coords=temp.coords,
            dims=temp.dims,
            attrs={"long_name": f"Standardized Temperature Index (STI-{window})"}
        )

    @staticmethod
    def heat_wave_index(
        temp: xr.DataArray,
        threshold_percentile: float = 90,
        window: int = 3
    ) -> xr.DataArray:
        """
        Calculate heat wave index (persistence of high temperatures).

        Parameters
        ----------
        temp : xr.DataArray
            Temperature data (time, lat, lon).
        threshold_percentile : float
            Percentile threshold for "hot" days (default: 90th).
        window : int
            Window for detecting consecutive hot days (default: 3).

        Returns
        -------
        xr.DataArray
            Heat wave intensity (0-1, where 1 is extreme heat).
        """
        # Calculate threshold (90th percentile)
        threshold = temp.quantile(threshold_percentile / 100)

        # Identify hot days
        hot_days = (temp > threshold).astype(float)

        # Calculate rolling sum (consecutive hot days)
        time, lat, lon = temp.shape
        hot_days_flat = hot_days.values.reshape(time, -1)

        hwi = np.zeros_like(hot_days_flat)

        for i in range(hot_days_flat.shape[1]):
            hwi[:, i] = pd.Series(hot_days_flat[:, i]).rolling(window, min_periods=1).sum()

        hwi = hwi.reshape(temp.shape)
        hwi = hwi / window  # Normalize to 0-1

        return xr.DataArray(
            hwi,
            coords=temp.coords,
            dims=temp.dims,
            attrs={
                "long_name": f"Heat Wave Index (>{threshold_percentile}th percentile, {window}-day window)",
                "units": "dimensionless (0-1)"
            }
        )

    @staticmethod
    def extreme_heat_days(
        temp: xr.DataArray,
        threshold: float
    ) -> xr.DataArray:
        """
        Identify days exceeding temperature threshold.

        Parameters
        ----------
        temp : xr.DataArray
            Temperature data (time, lat, lon).
        threshold : float
            Temperature threshold (°C).

        Returns
        -------
        xr.DataArray
            Binary array (1=extreme heat day, 0=normal).
        """
        extreme = (temp > threshold).astype(float)

        return extreme.assign_attrs({
            "long_name": f"Extreme Heat Days (T > {threshold}°C)",
            "units": "binary (0=no, 1=yes)"
        })


class CompoundIndices:
    """Calculate indices for compound drought and heatwave events."""

    @staticmethod
    def compound_event_index(
        drought_index: xr.DataArray,
        heat_index: xr.DataArray,
        drought_threshold: float = -0.5,
        heat_threshold: float = 0.5
    ) -> xr.DataArray:
        """
        Identify compound drought and heatwave events.

        Parameters
        ----------
        drought_index : xr.DataArray
            Drought index (e.g., SPI, standardized).
        heat_index : xr.DataArray
            Heat index (e.g., STI, standardized).
        drought_threshold : float
            Threshold for drought conditions (default: -0.5 std).
        heat_threshold : float
            Threshold for heat conditions (default: 0.5 std).

        Returns
        -------
        xr.DataArray
            Compound event index (0=neither, 1=drought, 2=heat, 3=both).
        """
        # Classify events
        drought = drought_index < drought_threshold
        heat = heat_index > heat_threshold

        # Combine: 0=neither, 1=drought only, 2=heat only, 3=both
        compound = (drought.astype(int) + heat.astype(int) * 2)

        return compound.assign_attrs({
            "long_name": "Compound Drought-Heatwave Event Index",
            "units": "0=none, 1=drought, 2=heat, 3=compound"
        })

    @staticmethod
    def cdhw_magnitude(
        drought_index: xr.DataArray,
        heat_index: xr.DataArray
    ) -> xr.DataArray:
        """
        Calculate magnitude of compound drought-heatwave events.

        Parameters
        ----------
        drought_index : xr.DataArray
            Drought index (negative values indicate drought).
        heat_index : xr.DataArray
            Heat index (positive values indicate heat).

        Returns
        -------
        xr.DataArray
            Magnitude of compound events (0-2, where 2 is most extreme).
        """
        # Normalize indices to 0-1 range
        drought_norm = np.abs(drought_index.clip(max=0)) / np.abs(drought_index.min())
        heat_norm = heat_index.clip(min=0) / heat_index.max()

        # Magnitude = average of normalized indices
        magnitude = (drought_norm + heat_norm) / 2

        return magnitude.assign_attrs({
            "long_name": "CDHW Event Magnitude",
            "units": "dimensionless (0-1)"
        })
