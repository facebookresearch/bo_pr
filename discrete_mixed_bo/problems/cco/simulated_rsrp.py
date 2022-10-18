#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import bisect
import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# System Constants
MIN_INTERFERENCE_POWER_WATT = 1e-24


class SimulatedRSRP:
    """Class that initializes RSRP and interference maps from disk.

    The data is provided in a particular format described below.

    Build this class from NPZ files. A single NPZ file consists of several lists and matrices:
            data['x'] - a list of x coordinates representing the x-axis of the map
            data['y'] - a list of y coordinates representing the y-axis of the map
            data['y'] - a fixed z coordinate of this map
            data['ptx'] - Transmit power of the sectors.
            data['Txnpwr'] - a 3D matix, which is the powermap of several sectors located in base station n
            data['Txnloc'] - The location of base station n

    A NPZ file is named as powermapDT{%d}.npz, where {%d} indicates the downtilt of
    all the base stations in this map.

    The API consists of the following methods:
        get_RSRP_and_interference_powermap:
                                This method returns the RSRP powermap, interference powermap and
                                connecting sectors map, given the configuration for each sector.
        get_RSRP_and_interference_for_location:
                                This method returns the RSRP, interference power and connecting sector idx
                                given a single location and the configuration.
        get_configuration_shape:
                                This method returns the configuration shape. Each configuration is
                                a 2-D Array with the shape [2, num_total_sectors]. Each column
                                contains the downtilt and transmit power, respectively, for different sectors.
        get_configuration_range:
                                This method will return the valid ranges for downtilts and transmit powers.
                                Both are specified as a minimum and maximum value.
        get_locations_range:
                                This method will return the range of the map.
                                Two Coordinate objects will be returned: xy_min, xy_max
                                Location(x,y) is the meters from the basestation center(0, 0).
                                The valid location will be:
                                xy_min.x <= location.x <= xy_max.y, xy_min.y <= location.y <= xy_max.y


    powermatrix path will be like:
        powermaps_path = "/mnt/shared/yuchenq/power_maps/*.npz"  # Now we have downtilt from 0 to 10, 11 files

    Sample Code:
        min_Tx_power_dBm, max_Tx_power_dBm = 30, 50
        simulated_rsrp = SimulatedRSRP.construct_from_npz_files(powermaps_path, (min_Tx_power_dBm, max_Tx_power_dBm))

        # Get configuration range for downtilts and powers
        (
            (min_downtilt, max_downtilt),
            (min_tx_power_dBm, max_tx_power_dBm),
        ) = self.simulated_rsrp.get_configuration_range()
        xy_min, xy_max = simulated_rsrp.get_locations_range()

        # Discretize power choices to integer values in range
        tx_power_choices = list(range(int(min_tx_power_dBm), int(max_tx_power_dBm + 1)))
        downtilts_choices = list(range(int(min_downtilt), int(max_downtilt + 1)))

        # Get the number of total sectors
        _, num_sectors = self.simulated_rsrp.get_configuration_shape()

        # Random configuration
        downtilts_for_sectors = np.random.choice(downtilts_choices, num_sectors)
        power_for_sectors = np.random.choice(tx_power_choices, num_sectors)
        configuration = (downtilts_for_sectors, power_for_sectors)

        # Get rsrp and interference powermap
        rsrp_powermap, interference_powermap, _ = simulated_rsrp.get_RSRP_and_interference_powermap(configurations)

        # Get rsrp and interference from location
        location = Coordinate(0, 0)
        rsrp, interference, _ = simulated_rsrp.get_RSRP_and_interference_for_location(location, configurations)
    """

    @dataclass
    class Coordinate:
        __slots__ = "x", "y"
        x: float
        y: float

    @dataclass
    class Powermap:
        """Dataclass to store a powermap with the specific downtilt of all base stations.

        power_matrix: 3D matrix
        base_station_locations: base station locations
        xy_min: minimum x and y
        xy_max: maximum x and y
        fixed_z: z value
        num_sectors_per_base_station: number of sectors of a single base station
        """

        power_matrix: np.ndarray
        base_station_locations: np.ndarray
        xy_min: "SimulatedRSRP.Coordinate"
        xy_max: "SimulatedRSRP.Coordinate"
        fixed_z: float
        num_sectors_per_base_station: List[int]

    @dataclass
    class Metadata:
        """Dataclass to store map metadata."""

        xy_min: "SimulatedRSRP.Coordinate"
        xy_max: "SimulatedRSRP.Coordinate"
        fixed_z: float
        resolution: float
        num_sectors_per_base_station: List[int]

    def __init__(
        self,
        powermaps: Dict[int, Any],
        min_TX_power_dBm: float,
        max_TX_power_dBm: float,
    ):
        self.downtilts_maps = {
            float(i): SimulatedRSRP.build_single_powermap(powermaps[i])
            for i in powermaps.keys()
        }
        # Get maps size, resolution and base stations distribution
        metadata = self.get_metadata(self.downtilts_maps)
        self.xy_min = metadata.xy_min
        self.xy_max = metadata.xy_max
        self.fixed_z = (metadata.fixed_z,)
        self.resolution = metadata.resolution
        self.num_sectors_per_base_station = metadata.num_sectors_per_base_station
        self.num_basestations = len(self.num_sectors_per_base_station)
        self.num_total_sectors = sum(self.num_sectors_per_base_station)
        self.min_TX_power_dBm = min_TX_power_dBm
        self.max_TX_power_dBm = max_TX_power_dBm

        self.downtilts_keys = np.asarray(list(self.downtilts_maps.keys()))
        self.downtilts_keys.sort()
        self.min_downtilt = self.downtilts_keys[0]
        self.max_downtilt = self.downtilts_keys[-1]

    def get_configuration_range(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return valid ranges of input configurations, later useful for
        `get_RSRP_and_interference_for_location`.

        This method returns the following 2-tuple:
            1. 2-tuple specifying (min downtilt, max downtilt) range
            2. 2-tuple specifying (min Tx power, max Tx power) range
        """
        return (
            (self.min_downtilt, self.max_downtilt),
            (self.min_TX_power_dBm, self.max_TX_power_dBm),
        )

    def get_locations_range(
        self,
    ) -> Tuple["SimulatedRSRP.Coordinate", "SimulatedRSRP.Coordinate"]:
        """Return the range of x,y in the map."""
        return self.xy_min, self.xy_max

    def get_configuration_shape(self) -> Tuple[int, int]:
        """Return the number of base stations and sectors.

        In tandem with `get_configuration_range`, this method is useful for
        constructing valid configuration inputs for the main API calls:
            1. `get_RSRP_and_interference_for_location`
            2. `get_RSRP_and_interference_powermap`
        """
        return (self.num_basestations, self.num_total_sectors)

    def get_basestation_and_sector_idx(
        self, flattened_sector_idx: int
    ) -> Tuple[int, int]:
        """Given the flattened sector id, return the base station idx and its sector idx"""
        if flattened_sector_idx >= self.num_total_sectors or flattened_sector_idx < 0:
            raise ValueError("flattened_sector_id is out of range")
        base_station_idx = 0
        sector_idx = 0
        while (
            sector_idx + self.num_sectors_per_base_station[base_station_idx]
        ) <= flattened_sector_idx:
            sector_idx += self.num_sectors_per_base_station[base_station_idx]
            base_station_idx += 1
        return base_station_idx, flattened_sector_idx - sector_idx

    @staticmethod
    def watt_to_dBm(x: Union[float, np.ndarray]) -> float:
        return 10 * np.log10(x) + 30

    @staticmethod
    def dBm_to_watt(x: Union[float, np.ndarray]) -> float:
        return 10 ** (x / 10.0 - 3)

    @staticmethod
    def get_nearest_discrete_downtilts(
        downtilts_keys: np.ndarray, downtilt: float
    ) -> Tuple[float, float]:
        """Return the nearest discrete downtilts for the given downtilt.

        downtilts_keys is the sorted numpy 1-D array, storing the discrete
        downtilts. Given downtilts, return the interpolation range:
                    [lower_downtilt, upper_downtilt]

        If downtilts_keys only contains one downtilt, the interpolation
        will not be needed. Such situation will be checked before this
        method is called in the API.
        """
        # Check the length
        if len(downtilts_keys) <= 1:
            raise ValueError("Can't do interpolation with only one discrete downtilt")

        # Check if downtilt is in the configiration range
        if downtilt > downtilts_keys[-1] or downtilt < downtilts_keys[0]:
            raise ValueError("Downtilt is out of the range")

        # Using bisect to find the nearest upper downtilt and lower downtilt indices
        upper_downtilt_idx = bisect.bisect(
            downtilts_keys, downtilt, hi=len(downtilts_keys) - 1
        )
        lower_downtilt_idx = upper_downtilt_idx - 1
        upper_downtilt = downtilts_keys[upper_downtilt_idx]
        lower_downtilt = downtilts_keys[lower_downtilt_idx]
        return (lower_downtilt, upper_downtilt)

    @staticmethod
    def get_resolution_from_powermap(
        powermap: "SimulatedRSRP.Powermap",
    ) -> "SimulatedRSRP.Coordinate":
        """Return the reslution of a Powermap Object.

        If the length of one axis of the 2-D map is 1, the resolution for this axis
        will be set as 1 for easy calculation of x-y index.
        """
        x_len, y_len, _ = powermap.power_matrix.shape
        resolution_x = (
            1 if x_len == 1 else (powermap.xy_max.x - powermap.xy_min.x) / (x_len - 1)
        )
        resolution_y = (
            1 if y_len == 1 else (powermap.xy_max.y - powermap.xy_min.y) / (y_len - 1)
        )
        return SimulatedRSRP.Coordinate(resolution_x, resolution_y)

    @staticmethod
    def get_xy_idx(
        resolution: "SimulatedRSRP.Coordinate",
        location: "SimulatedRSRP.Coordinate",
        xy_min: "SimulatedRSRP.Coordinate",
    ) -> Tuple[int, int]:
        """Return the x and y axis index, given the location and the resolution"""
        x_idx, y_idx = (
            int((location.x - xy_min.x) // resolution.x),
            int((location.y - xy_min.y) // resolution.y),
        )
        return (x_idx, y_idx)

    # Return RSRP of a single point
    def get_RSRP_and_interference_for_location(
        self,
        location: "SimulatedRSRP.Coordinate",
        configurations: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[float, float, int]:
        """Get RSRP and interference power, given the location and configuration.

        Configuration contains downtilts and transmit power for all the sectors.
        The first list in the configuration is for downtilts and
        the second is for transmit powers.

        The return values are RSRP, interference power and serving sector idx.
        """
        # Check if the location is in the resonable range
        if not (
            self.xy_min.x <= location.x <= self.xy_max.x
            and self.xy_min.y <= location.y <= self.xy_max.y
        ):
            raise ValueError("Current location is outside of the map!")

        # Check if the configurations format has the right shape
        try:
            assert len(configurations) == 2
            assert (
                len(configurations[0])
                == len(configurations[1])
                == self.num_total_sectors
            )
        except AssertionError:
            logging.error("Configurations shape doesn't fit")

        # Create an array to store the received powers from all the sectors
        rx_powers_dBm = np.zeros(self.num_total_sectors, dtype=np.float32)

        # Calculate RSRP, Interference power, and serving sector idx
        for i in range(self.num_total_sectors):
            # Get configurations, which are transmit power and downtilt
            configured_downtilt = configurations[0][i]
            configured_transmit_power_dBm = configurations[1][i]
            # Check if the configurations are in the right range
            if not (
                self.min_TX_power_dBm
                <= configured_transmit_power_dBm
                <= self.max_TX_power_dBm
            ):
                raise ValueError("Transmit Power is out of the range")

            # Get the resolution of the map, and calculate the idx of locations
            x_idx, y_idx = self.get_xy_idx(self.resolution, location, self.xy_min)
            # Get the received power from one sector
            received_power_dBm = (
                self.get_power_for_downtilt_sector(configured_downtilt, i, x_idx, y_idx)
                + configured_transmit_power_dBm
            )

            rx_powers_dBm[i] = received_power_dBm

        # Calculate RSRP and interference power
        # The maximum received power from all the sectors is
        # defined as the serving/attached cell RSRP.
        # The remaining received powers will be regarded as interference powers.
        rsrp_dBm, serving_sector_idx = np.max(rx_powers_dBm), np.argmax(rx_powers_dBm)
        interference_power_watt = sum(
            SimulatedRSRP.dBm_to_watt(
                rx_powers_dBm[np.arange(self.num_total_sectors) != serving_sector_idx]
            )
        )
        # Transfer to dbm
        interference_power_dBm = SimulatedRSRP.watt_to_dBm(interference_power_watt)

        return rsrp_dBm, interference_power_dBm, serving_sector_idx

    def get_RSRP_and_interference_powermap(
        self, configurations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns RSRP and interference power for all the locations in the map.

        `configurations` contains downtilts and transmit power for all the sectors.
        The first list in the configuration contains downtilts and
        the second contains transmit powers.

        Return 3 maps for:
            rsrp_powermap: RSRP power map
            interference_powermap: interference power map
            serving_sector_idx_map: conncecting sectors index map
        """
        # Check if the configurations format has the right shape
        try:
            assert len(configurations) == 2
            assert (
                len(configurations[0])
                == len(configurations[1])
                == self.num_total_sectors
            )
        except AssertionError:
            logging.error("Configurations shape doesn't fit")

        # Get downtilts and tx powers from configurations
        downtilts_for_sectors = configurations[0]
        tx_powers_for_sectors = configurations[1]

        # Check if the power configuration is right
        if (
            max(tx_powers_for_sectors) > self.max_TX_power_dBm
            or min(tx_powers_for_sectors) < self.min_TX_power_dBm
        ):
            raise ValueError("Transmit Power is out of the range")

        # Construct power matrices files from the configutration
        power_matrices = np.stack(
            [
                self.get_power_for_downtilt_sector(downtilt, flattened_sector_idx)
                + tx_powers_for_sectors[flattened_sector_idx]
                for flattened_sector_idx, downtilt in enumerate(downtilts_for_sectors)
            ],
            -1,
        )

        # Get RSRP powermap and serving sector idx map
        rsrp_power_map_dBm = np.amax(power_matrices, -1)
        serving_sector_idx_map = np.argmax(power_matrices, -1)

        # 1. Convert power from dBm to watt
        # 2. Sum the powers in every location and substract RSRP to get interference power
        # 3. Set minimum threshold 1e-24 to avoid 0 watt power
        interference_power_map_watt = np.maximum(
            MIN_INTERFERENCE_POWER_WATT,
            np.sum(SimulatedRSRP.dBm_to_watt(power_matrices), -1)
            - SimulatedRSRP.dBm_to_watt(rsrp_power_map_dBm),
        )

        # Get interference power map in dBm
        interference_power_map_dbm = SimulatedRSRP.watt_to_dBm(
            interference_power_map_watt
        )

        return (rsrp_power_map_dBm, interference_power_map_dbm, serving_sector_idx_map)

    def get_power_for_downtilt_sector(
        self,
        downtilt: float,
        flattened_sector_idx: int,
        x_idx: Optional[int] = None,
        y_idx: Optional[int] = None,
    ) -> Union[np.ndarray, float]:
        """Return interpolated power matrix or scalar power in given location

        If the x_idx and y_idx are given, the scalar power for this location
        will be calculated. Otherwise it will return the power matrix
        for the entire map.
        """
        # Check given x_idx and y_idx
        is_xy_given = x_idx is not None and y_idx is not None

        # Check if the interpolation is needed
        if downtilt in self.downtilts_maps:
            if is_xy_given:
                return self.downtilts_maps[downtilt].power_matrix[
                    x_idx, y_idx, flattened_sector_idx
                ]
            else:
                return self.downtilts_maps[downtilt].power_matrix[
                    :, :, flattened_sector_idx
                ]
        else:
            # Interpolation begin
            # 1. Find the nearest lower and upper downtilt
            lower_downtilt, upper_downtilt = self.get_nearest_discrete_downtilts(
                self.downtilts_keys, downtilt
            )

            # 2. Get the power matrix or scalar power for lower and upper downtilts
            # Check if x_idx and y_idx are given, calculate the scalar power
            if is_xy_given:
                upper_downtilt_power = self.downtilts_maps[upper_downtilt].power_matrix[
                    x_idx, y_idx, flattened_sector_idx
                ]
                lower_downtilt_power = self.downtilts_maps[lower_downtilt].power_matrix[
                    x_idx, y_idx, flattened_sector_idx
                ]
            # Otherwise get the power matrix
            else:
                upper_downtilt_power = self.downtilts_maps[upper_downtilt].power_matrix[
                    :, :, flattened_sector_idx
                ]
                lower_downtilt_power = self.downtilts_maps[lower_downtilt].power_matrix[
                    :, :, flattened_sector_idx
                ]

            # 3. Linear interpolation
            downtilt_power = (
                (upper_downtilt_power - lower_downtilt_power)
                / (upper_downtilt - lower_downtilt)
            ) * (downtilt - lower_downtilt) + lower_downtilt_power

            return downtilt_power

    @staticmethod
    def construct_from_npz_files(
        power_maps_path: str, power_range: Tuple[float, float]
    ) -> "SimulatedRSRP":
        """Construct power map data from multiple power maps (npz format).

        power_maps_path is filepath (local or mounted), e.g.
        "/mnt/shared/yuchenq/power_maps/*.npz".

        Power maps are loaded into the downtilts_maps dictionary,
        keyed on downtilts.

        npz files are generated from original JSON files. Here is the sample code:
            powermaps_dir = Path("/mnt/shared/yuchenq/power_maps")
            for fn in powermaps_dir.iterdir():
                npfn = fn.name.replace(".json", ".npz")
                with open(fn, "r") as f:
                    pmap = json.load(f)
                for k, vals in pmap.items():
                    pmap[k] = np.array(vals)
                np.savez_compressed(powermaps_dir.joinpath(npfn), **pmap)
        """
        downtilts_maps = {}
        power_maps_path = glob.glob(
            os.path.abspath(os.path.expandvars(power_maps_path))
        )
        for file_path in power_maps_path:
            try:
                downtilt = float(re.search(r"DT\d+", file_path).group()[2:])
            except AttributeError:
                logging.error("No downtilt parameter configuration find")
                raise
            if downtilt in downtilts_maps:
                logging.info("Duplicated downtilt %d files", downtilt)
            else:
                downtilts_maps[downtilt] = SimulatedRSRP.build_single_powermap(
                    np.load(file_path)
                )

        # Check if the map object has been successfully built
        if not downtilts_maps:
            logging.error("No power map files found!")
            raise

        # Construct the simulation object
        simulated_rsrp = SimulatedRSRP(
            downtilts_maps=downtilts_maps,
            min_TX_power_dBm=power_range[0],
            max_TX_power_dBm=power_range[1],
        )
        return simulated_rsrp

    @staticmethod
    def build_single_powermap(npz_data: Dict[str, Any]) -> "SimulatedRSRP.Powermap":
        """Construct a single power map for the specific downtilt.

        The power_matrix will have the following dimensions:
            [x, y, num_total_sectors]

        The power_matrix contains the received powers
        assuming 0 dBm transmit power.
        """

        x_coord = npz_data["x"]
        y_coord = npz_data["y"]
        z_coord = npz_data["z"]

        # Check the map if it is the uniform grid
        if not np.allclose(
            x_coord, np.linspace(x_coord[0], x_coord[-1], len(x_coord))
        ) or not np.allclose(
            y_coord, np.linspace(y_coord[0], y_coord[-1], len(y_coord))
        ):
            raise ValueError("xy 2D map must be uniform grid")

        # Transmit power, stored in Watt, converted to dBm
        TX_power_dBm = SimulatedRSRP.watt_to_dBm(npz_data["ptx"])

        # Try to get the number of base stations from the file
        try:
            num_base_stations = int(
                re.search(r"\d+", list(npz_data.keys())[-1]).group()
            )
        except ValueError:
            logging.error("Unable to determine the number of base stations")
            raise

        # Store recived powers from different locations
        rx_powers = []

        # Store number of sectors in different base stations
        num_sectors_per_base_station = []

        # Transmitter locations (transmitter, [x, y, z])
        base_station_locations = np.zeros((num_base_stations, 3))

        # Get the number of sectors of a single base station
        # The power map structure will be [x, y, num_total_sectors]
        for i in range(num_base_stations):
            label = "Tx{}".format(i + 1)
            num_sectors_per_base_station.append(len(npz_data[label + "pwr"][0][0]))
            rx_powers.append(npz_data[label + "pwr"] - TX_power_dBm)
            base_station_locations[i] = npz_data[label + "loc"]

        powermap = SimulatedRSRP.Powermap(
            power_matrix=np.concatenate(rx_powers, -1),
            base_station_locations=base_station_locations,
            xy_min=SimulatedRSRP.Coordinate(min(x_coord), min(y_coord)),
            xy_max=SimulatedRSRP.Coordinate(max(x_coord), max(y_coord)),
            fixed_z=z_coord,
            num_sectors_per_base_station=num_sectors_per_base_station,
        )
        return powermap

    @staticmethod
    def get_metadata(
        downtilts_maps: Dict[float, "SimulatedRSRP.Powermap"]
    ) -> "SimulatedRSRP.Metadata":
        """Analyze the maps and get information about sizes, resolution,
        and number of total sectors.

        Return xy_min, xy_max, fixed_z, num_total sectors and
        a list of num_sectors of each basetation.
        """
        xy_min = None
        xy_max = None
        fixed_z = None
        num_sectors_per_base_station = []
        for powermap in downtilts_maps.values():
            # We need guarantee all the maps in different files have the same range and resolutions
            if not xy_min:
                xy_min = powermap.xy_min
                xy_max = powermap.xy_max
                fixed_z = powermap.fixed_z
                resolution = SimulatedRSRP.get_resolution_from_powermap(powermap)
            else:
                try:
                    assert xy_min == powermap.xy_min
                    assert xy_max == powermap.xy_max
                    assert resolution == SimulatedRSRP.get_resolution_from_powermap(
                        powermap
                    )
                except AssertionError:
                    logging.error("Powermaps' sizes or resolutions don't match!")
                    raise

            # Check if the number of sectors math
            if not num_sectors_per_base_station:
                num_sectors_per_base_station = powermap.num_sectors_per_base_station
            else:
                try:
                    assert (
                        num_sectors_per_base_station
                        == powermap.num_sectors_per_base_station
                    )
                except AssertionError:
                    logging.error(
                        "Number of sectors for each base station doesn't match"
                    )
                    raise
        metadata = SimulatedRSRP.Metadata(
            xy_min, xy_max, fixed_z, resolution, num_sectors_per_base_station
        )
        return metadata
