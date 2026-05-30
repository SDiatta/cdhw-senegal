"""
Data Loader Module
Functions for reading and processing NetCDF files for CDHW analysis.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


class NetCDFLoader:
    """Load and process NetCDF files for climate data analysis."""

    def __init__(self, verbose: bool = True):
        """
        Initialize NetCDF Loader.

        Parameters
        ----------
        verbose : bool
            Whether to print information during data loading.
        """
        self.verbose = verbose
        self.datasets = {}

    def load_file(self, filepath: str, name: Optional[str] = None) -> xr.Dataset:
        """
        Load a single NetCDF file.

        Parameters
        ----------
        filepath : str
            Path to the NetCDF file.
        name : str, optional
            Name to store the dataset. If None, uses the filename.

        Returns
        -------
        xr.Dataset
            Loaded xarray Dataset.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if self.verbose:
            print(f"Loading: {filepath.name}")

        ds = xr.open_dataset(filepath)
        dataset_name = name or filepath.stem

        self.datasets[dataset_name] = ds

        if self.verbose:
            print(f"  Shape: {ds.dims}")
            print(f"  Variables: {list(ds.data_vars)}")

        return ds

    def load_directory(self, dirpath: str, pattern: str = "*.nc") -> Dict[str, xr.Dataset]:
        """
        Load all NetCDF files from a directory.

        Parameters
        ----------
        dirpath : str
            Path to directory containing NetCDF files.
        pattern : str
            File pattern to match (default: "*.nc").

        Returns
        -------
        dict
            Dictionary of loaded datasets.
        """
        dirpath = Path(dirpath)

        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")

        files = sorted(dirpath.glob(pattern))

        if not files:
            print(f"No files matching {pattern} found in {dirpath}")
            return {}

        if self.verbose:
            print(f"Found {len(files)} NetCDF file(s)")

        for filepath in files:
            self.load_file(filepath)

        return self.datasets

    def get_dataset(self, name: str) -> xr.Dataset:
        """Get a loaded dataset by name."""
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(self.datasets.keys())}")
        return self.datasets[name]

    def list_datasets(self) -> list:
        """Return list of loaded dataset names."""
        return list(self.datasets.keys())

    def inspect_dataset(self, name: str) -> None:
        """Print detailed information about a dataset."""
        ds = self.get_dataset(name)
        print(f"\n=== Dataset: {name} ===")
        print(f"Dimensions: {dict(ds.dims)}")
        print(f"\nCoordinates:\n{ds.coords}")
        print(f"\nData Variables:\n{ds.data_vars}")
        print(f"\nAttributes:\n{ds.attrs}")


class SenegalClimateLDataLoader(NetCDFLoader):
    """Specialized loader for Senegal climate data with precipitation and temperature."""

    def __init__(self, verbose: bool = True):
        """
        Initialize Senegal Climate Data Loader.

        Parameters
        ----------
        verbose : bool
            Whether to print information during data loading.
        """
        super().__init__(verbose=verbose)
        self.precip_ds = None
        self.temp_ds = None

    def load_precipitation_data(self, filepath: str) -> xr.Dataset:
        """
        Load precipitation data.

        Parameters
        ----------
        filepath : str
            Path to precipitation NetCDF file.

        Returns
        -------
        xr.Dataset
            Loaded precipitation dataset.
        """
        self.precip_ds = self.load_file(filepath, name="precipitation")
        return self.precip_ds

    def load_temperature_data(self, filepath: str) -> xr.Dataset:
        """
        Load temperature data.

        Parameters
        ----------
        filepath : str
            Path to temperature NetCDF file.

        Returns
        -------
        xr.Dataset
            Loaded temperature dataset.
        """
        self.temp_ds = self.load_file(filepath, name="temperature")
        return self.temp_ds

    def get_precipitation_variable(self, var_name: Optional[str] = None) -> xr.DataArray:
        """
        Get precipitation variable from dataset.

        Parameters
        ----------
        var_name : str, optional
            Name of precipitation variable. If None, attempts to auto-detect.

        Returns
        -------
        xr.DataArray
            Precipitation data array.
        """
        if self.precip_ds is None:
            raise ValueError("Precipitation data not loaded. Use load_precipitation_data() first.")

        if var_name:
            if var_name not in self.precip_ds.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in precipitation data.")
            return self.precip_ds[var_name]

        # Auto-detect precipitation variable
        common_precip_names = ["precip", "precipitation", "pr", "prate", "tp"]
        for name in common_precip_names:
            for var in self.precip_ds.data_vars:
                if name.lower() in var.lower():
                    if self.verbose:
                        print(f"Auto-detected precipitation variable: {var}")
                    return self.precip_ds[var]

        raise ValueError("Could not auto-detect precipitation variable.")

    def get_temperature_variable(self, var_name: Optional[str] = None) -> xr.DataArray:
        """
        Get temperature variable from dataset.

        Parameters
        ----------
        var_name : str, optional
            Name of temperature variable. If None, attempts to auto-detect.

        Returns
        -------
        xr.DataArray
            Temperature data array.
        """
        if self.temp_ds is None:
            raise ValueError("Temperature data not loaded. Use load_temperature_data() first.")

        if var_name:
            if var_name not in self.temp_ds.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in temperature data.")
            return self.temp_ds[var_name]

        # Auto-detect temperature variable
        common_temp_names = ["temp", "temperature", "tas", "t2m", "tmax", "tmin"]
        for name in common_temp_names:
            for var in self.temp_ds.data_vars:
                if name.lower() in var.lower():
                    if self.verbose:
                        print(f"Auto-detected temperature variable: {var}")
                    return self.temp_ds[var]

        raise ValueError("Could not auto-detect temperature variable.")

    def get_combined_data(
        self,
        precip_var: Optional[str] = None,
        temp_var: Optional[str] = None
    ) -> xr.Dataset:
        """
        Get combined precipitation and temperature dataset.

        Parameters
        ----------
        precip_var : str, optional
            Name of precipitation variable.
        temp_var : str, optional
            Name of temperature variable.

        Returns
        -------
        xr.Dataset
            Combined dataset with renamed variables.
        """
        precip = self.get_precipitation_variable(precip_var).rename("precipitation")
        temp = self.get_temperature_variable(temp_var).rename("temperature")

        # Ensure same spatial dimensions
        combined = xr.Dataset(
            {
                "precipitation": precip,
                "temperature": temp
            }
        )

        return combined
