"""
CDHW Analysis Module
Main analysis class for compound drought and heatwave event detection.
"""

import xarray as xr
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from .data_loader import SenegalClimateLDataLoader
from .indices import DroughtIndices, HeatIndices, CompoundIndices
from .visualization import CDHWVisualizer


class CDHWAnalyzer:
    """Main analyzer for compound drought and heatwave events."""

    def __init__(self, verbose: bool = True):
        """
        Initialize CDHW Analyzer.

        Parameters
        ----------
        verbose : bool
            Whether to print progress information.
        """
        self.verbose = verbose
        self.loader = SenegalClimateLDataLoader(verbose=verbose)
        self.visualizer = CDHWVisualizer()

        # Storage for data and indices
        self.precip = None
        self.temp = None
        self.indices = {}
        self.events = None

    def load_data(
        self,
        precip_file: str,
        temp_file: str,
        precip_var: Optional[str] = None,
        temp_var: Optional[str] = None
    ) -> None:
        """
        Load precipitation and temperature data.

        Parameters
        ----------
        precip_file : str
            Path to precipitation NetCDF file.
        temp_file : str
            Path to temperature NetCDF file.
        precip_var : str, optional
            Name of precipitation variable.
        temp_var : str, optional
            Name of temperature variable.
        """
        if self.verbose:
            print("Loading climate data...")

        self.loader.load_precipitation_data(precip_file)
        self.loader.load_temperature_data(temp_file)

        combined = self.loader.get_combined_data(precip_var, temp_var)

        self.precip = combined["precipitation"]
        self.temp = combined["temperature"]

        if self.verbose:
            print(f"  Precipitation shape: {self.precip.shape}")
            print(f"  Temperature shape: {self.temp.shape}")

    def calculate_drought_indices(
        self,
        methods: list = ["spi", "anomaly"],
        spi_window: int = 12
    ) -> Dict[str, xr.DataArray]:
        """
        Calculate drought indices.

        Parameters
        ----------
        methods : list
            Drought indices to calculate: 'spi', 'anomaly', 'smd'.
        spi_window : int
            Window size for SPI calculation.

        Returns
        -------
        dict
            Dictionary of drought indices.
        """
        if self.precip is None:
            raise ValueError("Precipitation data not loaded. Use load_data() first.")

        if self.verbose:
            print("Calculating drought indices...")

        drought_indices = {}

        if "spi" in methods:
            if self.verbose:
                print(f"  Calculating SPI (window={spi_window})...")
            drought_indices["spi"] = DroughtIndices.standardized_precipitation_index(
                self.precip,
                window=spi_window
            )

        if "anomaly" in methods:
            if self.verbose:
                print("  Calculating precipitation anomaly...")
            drought_indices["precip_anomaly"] = DroughtIndices.precipitation_anomaly(
                self.precip
            )

        if "smd" in methods:
            if self.verbose:
                print("  Calculating soil moisture deficit...")
            drought_indices["smd"] = DroughtIndices.soil_moisture_deficit(
                self.precip,
                self.temp
            )

        self.indices.update(drought_indices)
        return drought_indices

    def calculate_heat_indices(
        self,
        methods: list = ["sti", "anomaly", "hwi"],
        sti_window: int = 12,
        hwi_threshold_percentile: float = 90,
        hwi_window: int = 3
    ) -> Dict[str, xr.DataArray]:
        """
        Calculate heat indices.

        Parameters
        ----------
        methods : list
            Heat indices to calculate: 'sti', 'anomaly', 'hwi'.
        sti_window : int
            Window size for STI calculation.
        hwi_threshold_percentile : float
            Percentile for heat wave threshold.
        hwi_window : int
            Window size for heat wave detection.

        Returns
        -------
        dict
            Dictionary of heat indices.
        """
        if self.temp is None:
            raise ValueError("Temperature data not loaded. Use load_data() first.")

        if self.verbose:
            print("Calculating heat indices...")

        heat_indices = {}

        if "sti" in methods:
            if self.verbose:
                print(f"  Calculating STI (window={sti_window})...")
            heat_indices["sti"] = HeatIndices.standardized_temperature_index(
                self.temp,
                window=sti_window
            )

        if "anomaly" in methods:
            if self.verbose:
                print("  Calculating temperature anomaly...")
            heat_indices["temp_anomaly"] = HeatIndices.temperature_anomaly(
                self.temp
            )

        if "hwi" in methods:
            if self.verbose:
                print(f"  Calculating heat wave index...")
            heat_indices["hwi"] = HeatIndices.heat_wave_index(
                self.temp,
                threshold_percentile=hwi_threshold_percentile,
                window=hwi_window
            )

        self.indices.update(heat_indices)
        return heat_indices

    def calculate_compound_indices(
        self,
        drought_index: str = "spi",
        heat_index: str = "sti",
        drought_threshold: float = -0.5,
        heat_threshold: float = 0.5
    ) -> xr.DataArray:
        """
        Calculate compound drought-heatwave event index.

        Parameters
        ----------
        drought_index : str
            Name of drought index to use.
        heat_index : str
            Name of heat index to use.
        drought_threshold : float
            Threshold for drought conditions.
        heat_threshold : float
            Threshold for heat conditions.

        Returns
        -------
        xr.DataArray
            Compound event index.
        """
        if drought_index not in self.indices:
            raise ValueError(f"Drought index '{drought_index}' not calculated. "
                           f"Available: {list(self.indices.keys())}")
        if heat_index not in self.indices:
            raise ValueError(f"Heat index '{heat_index}' not calculated. "
                           f"Available: {list(self.indices.keys())}")

        if self.verbose:
            print(f"Calculating compound event index ({drought_index} vs {heat_index})...")

        drought = self.indices[drought_index]
        heat = self.indices[heat_index]

        self.events = CompoundIndices.compound_event_index(
            drought,
            heat,
            drought_threshold=drought_threshold,
            heat_threshold=heat_threshold
        )

        return self.events

    def get_event_statistics(self) -> pd.DataFrame:
        """
        Get statistics on compound events.

        Returns
        -------
        pd.DataFrame
            Statistics table.
        """
        if self.events is None:
            raise ValueError("Compound events not calculated. Use calculate_compound_indices() first.")

        # Count events
        event_counts = {}
        for event_type in [0, 1, 2, 3]:
            count = (self.events == event_type).sum()
            total_cells = self.events.size
            percent = (count / total_cells) * 100

            event_type_name = ["None", "Drought Only", "Heat Only", "Compound"][event_type]
            event_counts[event_type_name] = {
                "count": int(count.values),
                "percentage": float(percent.values)
            }

        return pd.DataFrame(event_counts).T

    def detect_event_periods(
        self,
        drought_index: str = "spi",
        heat_index: str = "sti",
        drought_threshold: float = -0.5,
        heat_threshold: float = 0.5,
        min_duration: int = 3
    ) -> pd.DataFrame:
        """
        Detect periods of compound drought-heatwave events.

        Parameters
        ----------
        drought_index : str
            Drought index name.
        heat_index : str
            Heat index name.
        drought_threshold : float
            Drought threshold.
        heat_threshold : float
            Heat threshold.
        min_duration : int
            Minimum event duration (months).

        Returns
        -------
        pd.DataFrame
            Event periods with start/end dates and intensity.
        """
        drought = self.indices[drought_index].mean(dim=["lat", "lon"])
        heat = self.indices[heat_index].mean(dim=["lat", "lon"])

        # Identify compound periods
        is_drought = drought < drought_threshold
        is_heat = heat > heat_threshold
        is_compound = is_drought & is_heat

        # Find event periods
        events_list = []
        in_event = False
        event_start = None
        event_values = []

        for time_idx, (t, is_event) in enumerate(zip(drought.time.values, is_compound.values)):
            if is_event and not in_event:
                # Start of event
                in_event = True
                event_start = t
                event_values = []

            if in_event:
                event_values.append({
                    "drought": drought.values[time_idx],
                    "heat": heat.values[time_idx]
                })

            if not is_event and in_event:
                # End of event
                if len(event_values) >= min_duration:
                    avg_drought = np.mean([v["drought"] for v in event_values])
                    avg_heat = np.mean([v["heat"] for v in event_values])

                    events_list.append({
                        "start_date": pd.Timestamp(event_start),
                        "end_date": pd.Timestamp(t),
                        "duration_months": len(event_values),
                        "avg_drought_index": avg_drought,
                        "avg_heat_index": avg_heat,
                        "intensity": (abs(avg_drought) + avg_heat) / 2
                    })

                in_event = False
                event_values = []

        return pd.DataFrame(events_list)

    def export_indices(self, output_dir: str) -> None:
        """
        Export calculated indices to NetCDF files.

        Parameters
        ----------
        output_dir : str
            Directory to save output files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Exporting indices to {output_dir}...")

        for name, data in self.indices.items():
            filepath = output_dir / f"{name}.nc"
            data.to_netcdf(filepath)

            if self.verbose:
                print(f"  Saved: {filepath}")

        if self.events is not None:
            filepath = output_dir / "compound_events.nc"
            self.events.to_netcdf(filepath)

            if self.verbose:
                print(f"  Saved: {filepath}")

    def get_summary(self) -> str:
        """
        Get summary of analysis results.

        Returns
        -------
        str
            Summary text.
        """
        summary = "=== CDHW Analysis Summary ===\n"

        if self.precip is not None:
            summary += f"\nPrecipitation Data:\n"
            summary += f"  Shape: {self.precip.shape}\n"
            summary += f"  Time range: {self.precip.time.values[0]} to {self.precip.time.values[-1]}\n"

        if self.temp is not None:
            summary += f"\nTemperature Data:\n"
            summary += f"  Shape: {self.temp.shape}\n"

        summary += f"\nCalculated Indices:\n"
        for name in self.indices.keys():
            summary += f"  - {name}\n"

        if self.events is not None:
            summary += f"\nCompound Events Summary:\n"
            stats = self.get_event_statistics()
            summary += str(stats)

        return summary
