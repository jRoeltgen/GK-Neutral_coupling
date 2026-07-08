import numpy as np
from pathlib import Path
import re
from neutral_coupling.common.eirene_io import eirene
from neutral_coupling.genex_coupling.eirene_interface import status
from neutral_coupling.common import triangle_mesh as triangles
from neutral_coupling.common import b2_io as B2IO
from scipy.constants import elementary_charge
from neutral_coupling.common.eirene_input_parser import EireneInputParser
from neutral_coupling.common.temperature_mapping_utils import default_sparse_temperature_handler


COLLISION_TEMPERATURE_CONFIG = {
    "atom-plasma": {
        "prefix": "Tn_",
        "pseudo": "Tn_ATOMS",
        "fort": "fort.105",
    },
    "molecule-plasma": {
        "prefix": "Tm_",
        "pseudo": "Tm_MOLECULES",
        "fort": "fort.115",
    },
    "testion-plasma": {
        "prefix": "Tti_",
        "pseudo": "Tti_TEST IONS",
        "fort": "fort.125",
    },
    "plasma-plasma": {
        "prefix": "Ti_",
        "pseudo": None,
        "fort": "fort.145",
    },
}
DEFAULT_TEMPERATURE_COLLISIONS = tuple(COLLISION_TEMPERATURE_CONFIG)


def requested_strata_output_order(requested_strata):
    """Return the normalized order expected in EIRENE diagnostic files."""
    numbered = sorted(
        stratum for stratum in requested_strata if stratum != "SUM"
    )
    if "SUM" in requested_strata:
        numbered.append("SUM")
    return numbered


def validate_temperature_collisions(collision_types):
    collision_types = tuple(collision_types)
    unknown = set(collision_types) - set(COLLISION_TEMPERATURE_CONFIG)
    if unknown:
        raise ValueError(f"Unknown checked collision types: {sorted(unknown)}")
    if len(collision_types) < 2:
        raise ValueError("Multiple-temperature checks require at least two collisions")
    if len(set(collision_types)) != len(collision_types):
        raise ValueError("Checked collision types must be unique")
    return collision_types


def checked_collision_mappers(collision_types):
    """Build strict test mappers for the selected temperature prefixes."""
    collision_types = validate_temperature_collisions(collision_types)
    mappers = {}

    for collision in collision_types:
        prefix = COLLISION_TEMPERATURE_CONFIG[collision]["prefix"]
        pseudo = COLLISION_TEMPERATURE_CONFIG[collision]["pseudo"]

        def mapper(
            mom,
            species,
            strata_dict,
            temperature_values,
            context,
            *,
            _prefix=prefix,
            _pseudo=pseudo,
            _collision=collision,
        ):
            if "SUM" not in strata_dict:
                raise ValueError(
                    f"Checked {_collision} source for '{species}' has no SUM "
                    f"stratum; available strata: {list(strata_dict)}"
                )
            labels = [
                label
                for label in temperature_values
                if label.startswith(_prefix)
                and label != _pseudo
            ]
            if len(labels) != 1:
                raise ValueError(
                    f"Expected exactly one '{_prefix}' temperature for the "
                    f"checked run, found {labels}"
                )
            return {
                "values": {labels[0]: strata_dict["SUM"]},
                "conserved_and_constrained": True,
            }

        mappers[collision] = mapper

    return mappers


def checked_sparse_temperature_handler(
    temperature,
    density,
    triangle_points,
    *,
    threshold,
):
    """Fill any sparse pattern; fail if no valid temperature exists."""
    handled = default_sparse_temperature_handler(
        temperature,
        density,
        triangle_points,
        threshold=1.0,
    )
    if handled is None:
        raise ValueError("No positive-density cells for checked temperature")
    return handled

class CheckedRunEirene:
    def __init__(self, mode="summed", collision_types=None):
        if mode not in {"summed", "multiple_temperatures"}:
            raise ValueError(f"Unknown CheckedRunEirene mode: {mode}")
        self.mode = mode
        if mode == "multiple_temperatures":
            self.collision_types = validate_temperature_collisions(
                collision_types or DEFAULT_TEMPERATURE_COLLISIONS
            )
        else:
            self.collision_types = ()
        self.edat = eirene()
        self.prev_norm = None
        self.history = []
        self.location_history = {}
        self.source_metadata = {}
        self.current_density_flat = None
        self.pending_source_response = None
        self.source_response_history = []
        self.constant_magnitude = None
        self.response_check_start = 2
        self.response_check_number = 0
        self.checked_delta_signs = []
        self.checked_density_deltas = []
        self.min_positive_support_cells = 3
        self.min_positive_support_weight_fraction = 0.5
        self.source_sign_buffer = 2
        self.source_sign_blocks = (1, -1, 1, 1)
        self.source_sign_block_length = None
        self.source_index = 0

    def __call__(self, timeout, eirene_path, command=None):
        # --- Read current GENE-X state (what EIRENE would see) ---
        b2dat = B2IO.B2(eirene_path)
        nx, ny = b2dat.gmtry["vol"].shape
        parser = EireneInputParser(eirene_path / "input.dat")
        parser.parse_requested_strata()
        parser.parse_species()
        ns = len(parser.species["bulk_ions"])
        self.edat.read_ft31(
            eirene_path / Path("fort.31"),
            nx,
            ny,
            ns,
        )
        self.edat.triangle_mesh = triangles.triangle_mesh(eirene_path)
        self.edat.triangle_mesh.calc_incenter()
        n = self.edat.fort31["na"]

        # --- Density checks (this is your GENE-X validation) ---
        assert np.isfinite(n).all(), "NaNs in GENE-X density"

        norm = np.linalg.norm(n)
        self.history.append(norm)
        self.current_density_flat = np.asarray(n).reshape(-1).astype(float)

        if self.prev_norm is not None:
            # ensure density is evolving
            if np.isclose(norm, self.prev_norm, rtol=1e-8):
                raise AssertionError("GENE-X density is not evolving between iterations")

        self.prev_norm = norm
        self._check_pending_source_response(self.current_density_flat)

        if self.mode == "multiple_temperatures":
            self._write_multiple_temperature_sources(
                eirene_path, b2dat, n
            )
            return status.SUCCESS

        # --- Build synthetic EIRENE response ---
        R = self.edat.triangle_mesh.incenter[:,0]
        Z = self.edat.triangle_mesh.incenter[:,1]

        source_index = self.source_index
        sign = self._summed_source_sign(source_index, eirene_path)

        R0, Z0 = 2.26, 0.0
        sigmasq_z = 0.01
        sigmasq_r = 0.00003
        dt = 1e-7

        r_plasma = np.mean(b2dat.gmtry["crx"],axis=2)
        z_plasma = np.mean(b2dat.gmtry["cry"],axis=2)
        plasma_dist = ((r_plasma - R0)**2 + (z_plasma - Z0)**2).ravel()
        plasma_source_idx = np.argmin(plasma_dist)
        source_density = float(n.ravel()[plasma_source_idx])
        if self.constant_magnitude == None:
            amplitude = source_density*10
            self.constant_magnitude = amplitude
        else:
            amplitude = self.constant_magnitude

        gaussian = np.exp(
            -((R - R0)**2) / sigmasq_r
            -((Z - Z0)**2) / sigmasq_z
        )
        gaussian /= np.max(gaussian)

        unit_conversion = elementary_charge/1e6
        source = amplitude * gaussian * sign / dt
        source *= unit_conversion
        self._record_expected_source_response(
            n,
            r_plasma,
            z_plasma,
            R0,
            Z0,
            sigmasq_r,
            sigmasq_z,
            sign,
            source_index,
        )

        # --- Source sanity checks ---
        assert np.isfinite(source).all(), "NaNs in synthetic EIRENE source"
        assert not np.all(source == 0), "Zero source generated"

        # --- Spatial structure checks ---
        idx_max = np.argmax(np.abs(source))

        # Check peak location
        dist = np.sqrt((R - R0)**2 + (Z - Z0)**2)
        closest_idx = np.argmin(dist)

        assert idx_max == closest_idx or dist[idx_max] < 2 * np.min(dist[dist >= 0])

        # Check decay: most of domain near zero
        threshold = 1e-5 * np.max(np.abs(source))
        fraction_small = np.mean(np.abs(source) < threshold)

        assert fraction_small > 0.9, "Source not sufficiently localized"

        # --- Write fort files ---
        output_strata = requested_strata_output_order(
            parser.requested_strata
        )
        strata_values = [
            source if stratum == "SUM" else np.zeros_like(source)
            for stratum in output_strata
        ]
        self.write_fort_source(
            eirene_path / "fort.100",
            source,
            parser.species["electrons"][0],
            strata_values=strata_values,
        )
        self.write_fort_source(
            eirene_path / "fort.105",
            source,
            parser.species["bulk_ions"][0],
            strata_values=strata_values,
        )
        self.source_index += 1

        return status.SUCCESS

    def _record_expected_source_response(
        self,
        density,
        r_plasma,
        z_plasma,
        R0,
        Z0,
        sigmasq_r,
        sigmasq_z,
        source_sign,
        source_index,
    ):
        density_flat = np.asarray(density, dtype=float).reshape(-1)
        source_shape = np.exp(
            -((np.asarray(r_plasma).reshape(-1) - R0) ** 2) / sigmasq_r
            -((np.asarray(z_plasma).reshape(-1) - Z0) ** 2) / sigmasq_z
        )
        if source_shape.shape != density_flat.shape:
            raise AssertionError(
                "Source response check shape mismatch on EIRENE grid: "
                f"source shape {source_shape.shape}, density shape "
                f"{density_flat.shape}"
            )
        if not np.isfinite(source_shape).all():
            raise AssertionError("Non-finite source footprint in response check")
        if np.all(source_shape == 0):
            raise AssertionError("Zero source footprint in response check")

        support = source_shape >= 0.1 * np.max(source_shape)
        if np.count_nonzero(support) < 3:
            support = source_shape > 0
        weights = np.zeros_like(source_shape, dtype=float)
        weights[support] = source_shape[support]
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            raise AssertionError("Empty source support in response check")
        weights /= weight_sum
        positive_density = density_flat > 0
        positive_support = support & positive_density
        positive_support_cells = int(np.count_nonzero(positive_support))
        positive_support_weight_fraction = float(np.sum(weights[positive_density]))
        if positive_support_cells < self.min_positive_support_cells:
            raise AssertionError(
                "Source response footprint does not overlap enough "
                "positive-density plasma cells: "
                f"{positive_support_cells} positive cells in support"
            )
        if (
            positive_support_weight_fraction
            < self.min_positive_support_weight_fraction
        ):
            raise AssertionError(
                "Source response footprint is mostly outside positive-density "
                "plasma cells: "
                f"positive weight fraction={positive_support_weight_fraction}"
            )

        if source_sign == 0:
            raise AssertionError("Ambiguous source sign in response check")

        baseline_density = float(np.sum(density_flat * weights))
        if baseline_density <= 0:
            raise AssertionError(
                "Source response baseline density is not positive: "
                f"{baseline_density}"
            )
        self.pending_source_response = {
            "label": f"source centered at R={R0}, Z={Z0}",
            "source_index": int(source_index),
            "R": float(R0),
            "Z": float(Z0),
            "weights": weights,
            "sign": float(source_sign),
            "baseline_density": baseline_density,
            "support_cells": int(np.count_nonzero(support)),
            "positive_support_cells": positive_support_cells,
            "positive_support_weight_fraction": positive_support_weight_fraction,
        }
        print(
            "recorded source response check:",
            "source index=", source_index,
            "label=", self.pending_source_response["label"],
            "baseline=", baseline_density,
            "source sign=", source_sign,
            "support cells=", np.count_nonzero(support),
            "positive support cells=", positive_support_cells,
            "positive support weight fraction=",
            positive_support_weight_fraction,
            flush=True,
        )

    def _check_pending_source_response(self, density_flat):
        if self.pending_source_response is None:
            return

        pending = self.pending_source_response
        weights = pending["weights"]
        current_density = float(np.sum(density_flat * weights))
        previous_density = pending["baseline_density"]
        density_delta = current_density - previous_density
        expected_sign = pending["sign"]
        scale = max(abs(current_density), abs(previous_density), 1.0)
        tolerance = 1e-8 * scale
        delta_sign = 0.0 if abs(density_delta) <= tolerance else float(np.sign(density_delta))
        self.source_response_history.append(
            {
                "label": pending["label"],
                "source_index": pending["source_index"],
                "previous_density": previous_density,
                "current_density": current_density,
                "density_delta": density_delta,
                "density_delta_sign": delta_sign,
                "source_sign": expected_sign,
                "support_cells": pending["support_cells"],
            }
        )
        print(
            "weighted density response:",
            "source index=", pending["source_index"],
            "label=", pending["label"],
            "previous=", previous_density,
            "current=", current_density,
            "delta=", density_delta,
            "delta sign=", delta_sign,
            "source sign=", expected_sign,
            "support cells=", pending["support_cells"],
            flush=True,
        )
        self.pending_source_response = None
        if self.response_check_number >= self.response_check_start:
            assert abs(density_delta) > tolerance, (
                "GENE-X density did not measurably respond over the source "
                "footprint. "
                f"Source={pending['label']}, previous density={previous_density}, "
                f"current density={current_density}, delta={density_delta}, "
                f"source sign={expected_sign}, "
                f"support cells={pending['support_cells']}"
            )
            self.checked_delta_signs.append(delta_sign)
            self.checked_density_deltas.append(density_delta)
        self.response_check_number += 1

    def assert_summed_source_updates_used(self, eirene_path=None):
        if self.mode != "summed":
            return

        if len(self.checked_delta_signs) < 2:
            raise AssertionError(
                "Not enough checked density responses to verify source updates: "
                f"{self.checked_delta_signs}"
            )

        sign_changes = sum(
            current != previous
            for previous, current in zip(
                self.checked_delta_signs[:-1],
                self.checked_delta_signs[1:],
            )
        )
        theoretical_updates = self._theoretical_source_updates(eirene_path)
        minimum_sign_changes = self._minimum_required_delta_sign_changes(
            theoretical_updates
        )
        print(
            "summed source response sign-change summary:",
            "checked signs=", self.checked_delta_signs,
            "sign changes=", sign_changes,
            "theoretical source updates=", theoretical_updates,
            "minimum required sign changes=", minimum_sign_changes,
            flush=True,
        )
        assert sign_changes >= minimum_sign_changes, (
            "GENE-X density response did not show enough sign changes for the "
            "block-sign source updates. "
            f"checked signs={self.checked_delta_signs}, "
            f"deltas={self.checked_density_deltas}, "
            f"sign changes={sign_changes}, "
            f"theoretical source updates={theoretical_updates}, "
            f"minimum required sign changes={minimum_sign_changes}"
        )

    def _minimum_required_delta_sign_changes(self, theoretical_updates):
        return 2

    def _summed_source_sign(self, source_index, eirene_path):
        block_length = self._summed_source_sign_block_length(eirene_path)
        if source_index < self.source_sign_buffer:
            return 1

        block_index = (source_index - self.source_sign_buffer) // block_length
        block_index = min(block_index, len(self.source_sign_blocks) - 1)
        return self.source_sign_blocks[block_index]

    def _summed_source_sign_block_length(self, eirene_path):
        if self.source_sign_block_length is not None:
            return self.source_sign_block_length

        theoretical_updates = self._theoretical_source_updates(eirene_path)
        if theoretical_updates is None:
            block_length = 3
        else:
            block_length = int(
                (theoretical_updates - self.source_sign_buffer)
                / len(self.source_sign_blocks)
            )
            block_length = max(1, block_length)

        self.source_sign_block_length = block_length
        print(
            "summed source sign pattern:",
            "initial positive buffer=", self.source_sign_buffer,
            "source sign blocks=", self.source_sign_blocks,
            "block length=", block_length,
            "theoretical source updates=", theoretical_updates,
            flush=True,
        )
        return block_length

    def _theoretical_source_updates(self, eirene_path):
        params = self._read_minimal_genex_params(eirene_path)
        if params is None:
            return None

        dt = params.get("dt")
        n_timesteps = params.get("n_timesteps")
        update_delta_t = params.get("update_delta_t")
        if dt is None or n_timesteps is None or update_delta_t is None:
            return None
        if update_delta_t <= 0:
            return None
        return dt * n_timesteps / update_delta_t

    def _read_minimal_genex_params(self, eirene_path):
        if eirene_path is None:
            return None

        eirene_path = Path(eirene_path)
        candidates = [
            eirene_path / "params_summed_temp.txt",
            eirene_path / "params_in.txt",
        ]
        for path in candidates:
            if path.exists():
                return self._parse_minimal_genex_params(path)
        return None

    def _parse_minimal_genex_params(self, path):
        params = {}
        pattern = re.compile(
            r"^\s*(dt|n_timesteps|update_delta_t)\s*=\s*([^!\s]+)"
        )
        for line in Path(path).read_text().splitlines():
            match = pattern.match(line)
            if not match:
                continue
            key, value = match.groups()
            value = value.replace("D", "e").replace("d", "e")
            if key == "n_timesteps":
                params[key] = int(float(value))
            else:
                params[key] = float(value)
        return params

    def _write_multiple_temperature_sources(self, eirene_path, b2dat, density):
        R = self.edat.triangle_mesh.incenter[:, 0]
        Z = self.edat.triangle_mesh.incenter[:, 1]
        r_plasma = np.mean(b2dat.gmtry["crx"], axis=2).ravel()
        z_plasma = np.mean(b2dat.gmtry["cry"], axis=2).ravel()
        plasma_points = np.column_stack([r_plasma, z_plasma])
        density_flat = np.asarray(density).reshape(-1)
        valid = (
            np.isfinite(plasma_points).all(axis=1)
            & np.isfinite(density_flat)
            & (density_flat > 0)
        )
        if np.count_nonzero(valid) < len(self.collision_types):
            raise ValueError(
                "Not enough positive-density plasma cells for distinct "
                "checked source locations"
            )
        valid_indices = np.flatnonzero(valid)
        valid_points = plasma_points[valid]

        centered = valid_points - valid_points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projection = centered @ vh[0]
        quantiles = np.linspace(0.15, 0.85, len(self.collision_types))
        target_projection = np.quantile(projection, quantiles)
        valid_center_indices = [
            int(np.argmin(np.abs(projection - target)))
            for target in target_projection
        ]
        center_indices = valid_indices[valid_center_indices]
        centers = plasma_points[center_indices]
        if len(np.unique(center_indices)) != len(center_indices):
            raise AssertionError("Could not select distinct source centers")

        separations = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )
        positive_separations = separations[separations > 0]
        minimum_separation = np.min(positive_separations)
        widths = np.linspace(
            minimum_separation / 10.0,
            minimum_separation / 5.0,
            len(self.collision_types),
        )

        parser = EireneInputParser(eirene_path / "input.dat")
        parser.parse_requested_strata()
        parser.parse_species()
        bulk_species = parser.species["bulk_ions"]
        if len(bulk_species) != 1:
            raise ValueError(
                "Checked multiple-temperature mode requires one bulk ion; "
                f"found {bulk_species}"
            )
        species_name = bulk_species[0]
        self._write_checked_fort46(
            eirene_path / "fort.46",
            parser,
            R,
            Z,
        )

        selected_forts = {
            COLLISION_TEMPERATURE_CONFIG[collision]["fort"]
            for collision in self.collision_types
        }
        for config in COLLISION_TEMPERATURE_CONFIG.values():
            if config["fort"] not in selected_forts:
                (eirene_path / config["fort"]).unlink(missing_ok=True)

        source_sign = 1.0
        unit_conversion = elementary_charge / 1e6
        dt = 1e-7

        for index, (collision, center, width, plasma_index) in enumerate(
            zip(self.collision_types, centers, widths, center_indices)
        ):
            gaussian = np.exp(
                -(
                    (R - center[0]) ** 2
                    + (Z - center[1]) ** 2
                )
                / width ** 2
            )
            gaussian /= np.max(gaussian)
            local_density = density_flat[plasma_index]
            amplitude = local_density * (index + 1) * 10.0
            source = (
                amplitude * gaussian * source_sign / dt * unit_conversion
            )
            assert np.isfinite(source).all()
            assert np.any(source != 0)

            history = self.location_history.setdefault(collision, [])
            history.append(float(local_density))
            self.source_metadata[collision] = {
                "center": center.copy(),
                "width": float(width),
                "triangle_peak": int(np.argmax(np.abs(source))),
                "plasma_index": plasma_index,
            }
            filename = COLLISION_TEMPERATURE_CONFIG[collision]["fort"]
            output_strata = requested_strata_output_order(
                parser.requested_strata
            )
            strata_values = [
                source if stratum == "SUM" else np.zeros_like(source)
                for stratum in output_strata
            ]
            self.write_fort_source(
                eirene_path / filename,
                source,
                species_name,
                collision=collision,
                strata_values=strata_values,
            )

    def _write_checked_fort46(self, filepath, parser, R, Z):
        """Write deterministic, spatially varying neutral temperatures."""
        ntri = R.size
        radial = np.divide(
            R - np.min(R),
            np.ptp(R),
            out=np.zeros_like(R),
            where=np.ptp(R) > 0,
        )
        vertical = np.divide(
            Z - np.min(Z),
            np.ptp(Z),
            out=np.zeros_like(Z),
            where=np.ptp(Z) > 0,
        )
        coordinate = radial + 0.5 * vertical
        class_data = {
            "a": ("atoms", "atom labels", 10.0),
            "m": ("molecules", "molecule labels", 30.0),
            "i": ("test_ions", "ion labels", 50.0),
        }

        fort46 = {
            "ntri": ntri,
            "ver": 20170930,
            "label": "CHECKED",
            "volumes": np.ones((ntri, 1)),
            "pux": np.zeros((ntri, 1)),
            "puy": np.zeros((ntri, 1)),
            "pvx": np.zeros((ntri, 1)),
            "pvy": np.zeros((ntri, 1)),
        }
        for key, (particle_class, labels_key, offset) in class_data.items():
            labels = parser.species[particle_class]
            count = len(labels)
            fort46[labels_key] = [f"{label}\n" for label in labels]
            fort46[{"a": "natm", "m": "nmol", "i": "nion"}[key]] = count
            density = np.empty((ntri, count))
            energy = np.empty((ntri, count))
            for index in range(count):
                density[:, index] = 1e17 * (
                    1.0 + 0.1 * coordinate + 0.05 * index
                )
                temperature = (
                    offset + 5.0 * (index + 1) * coordinate
                ) * elementary_charge
                energy[:, index] = density[:, index] * temperature
            fort46["pden" + key] = density
            fort46["eden" + key] = energy
            for component in ("vx", "vy", "vz"):
                fort46[component + "den" + key] = np.zeros(
                    (ntri, count)
                )

        self.edat.fort46 = fort46
        self.edat.write_ft46(filepath)

    def write_fort_source(
        self,
        filepath,
        values,
        species_name,
        collision="atom-plasma",
        strata_values=None,
    ):
        """
        values: array of length Ntri (triangle-ordered)
        species_name: "ELECTRONS" or ion name
        """
        if strata_values is None:
            strata_values = [values]

        with open(filepath, "w") as f:
            for block_values in strata_values:
                Ntri = len(block_values)
                # Header
                f.write(" ========================================================================\n")
                f.write(" ========================================================================\n")
                f.write(
                    f" PARTICLE SOURCE FROM {collision.upper()} INTERACTION\n"
                )
                f.write(f" {species_name}\n")
                f.write(" AMP*CM**-3\n")
                f.write(" ========================================================================\n")
                f.write(" ========================================================================\n")
                f.write(
                    f"{Ntri+1:10d}           1           1           1"
                    f"{Ntri+1:10d}\n"
                )

                # Body
                for i, val in enumerate(block_values, start=1):
                    f.write(f"{i:10d}    0    {val: .6E}\n")

                # Footer
                f.write(
                    "================================================"
                    "========================\n"
                )
                f.write(" AVERAGE VALUE   0.0000E+00\n")
                f.write(
                    "================================================"
                    "========================\n"
                )
                f.write(" ADDITIONAL CELLS\n")
                f.write(
                    "================================================"
                    "========================\n"
                )
