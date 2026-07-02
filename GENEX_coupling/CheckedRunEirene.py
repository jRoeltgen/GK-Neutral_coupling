import numpy as np
from pathlib import Path
from eireneIO import eirene
from eirene_interface import status
import triangle_mesh as triangles
import B2IO
from scipy.constants import elementary_charge
from EireneInputParser import EireneInputParser
from temperature_mapping_utils import default_sparse_temperature_handler


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
        self.counter = 1  # for oscillation

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

        if self.prev_norm is not None:
            # ensure density is evolving
            if np.isclose(norm, self.prev_norm, rtol=1e-8):
                raise AssertionError("GENE-X density is not evolving between iterations")

        self.prev_norm = norm

        if self.mode == "multiple_temperatures":
            self._write_multiple_temperature_sources(
                eirene_path, b2dat, n
            )
            return status.SUCCESS

        # --- Build synthetic EIRENE response ---
        R = self.edat.triangle_mesh.incenter[:,0]
        Z = self.edat.triangle_mesh.incenter[:,1]

        self.counter *= -1

        #R0, Z0 = 2.272, 0.0
        R0, Z0 = 2.26, 0.0
        sigmasq_z = 0.01
        sigmasq_r = 0.00003
        dt = 1e-7

        r_plasma = np.mean(b2dat.gmtry["crx"],axis=2)
        z_plasma = np.mean(b2dat.gmtry["cry"],axis=2)
        dist = ((r_plasma - R0)**2 + (z_plasma - Z0)**2).ravel()
        closest_idx = np.argmin(dist)
        amplitude = n.ravel()[closest_idx]*10
        print("Source amplitude=",amplitude)

        gaussian = np.exp(
            -((R - R0)**2) / sigmasq_r
            -((Z - Z0)**2) / sigmasq_z
        )
        gaussian /= np.max(gaussian)

        unit_conversion = elementary_charge/1e6
        source = amplitude * gaussian * self.counter / dt
        print("density=",np.max(n))
        print("Gaussian max",np.max(np.abs(gaussian)))
        print("counter=",self.counter)
        print("dt=",dt)
        print("source max amp (#/m^3s)",np.max(np.abs(source)))
        source *= unit_conversion
        print("source max amp (amp/cm^3s)",np.max(np.abs(source)))
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

        return status.SUCCESS

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

        self.counter *= -1
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
                amplitude * gaussian * self.counter / dt * unit_conversion
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
