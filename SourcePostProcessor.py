from collections import defaultdict
from temperature_mapping_utils import default_D_only_collision_mappers
import numpy as np

class SourcePostProcessor:

    def __init__(self, sources, temperature_values,
                 collision_mappers=None):
        """
        Converts:
            sources[moment][collision][species][stratum]
        into:
            sources_by_temperature[moment][species][T_label]

        temperature_values:
            dict: temperature label -> value
            e.g. {"Tn_D": array, "Ti_D+": array}

        collision_mappers:
            None selects the built-in D-only mapping. Any explicitly supplied
            mapping is trusted as the caller's intended decomposition.
        """
        self.sources = sources
        self.temperature_values = temperature_values
        self.using_default_collision_mappers = collision_mappers is None
        self.collision_mappers = (
            default_D_only_collision_mappers
            if collision_mappers is None
            else collision_mappers
        )
        if self.using_default_collision_mappers:
            self._validate_default_mapper_configuration()

        self.sources_by_species = None
        self.sources_by_temperature = None
        self.temperature = None

    # -----------------------------------
    # Step 1: transpose hierarchy
    # -----------------------------------
    def transpose_sources(self):
        """
        sources[moment][collision][species][stratum]
        -> sources_T[moment][species][collision][stratum]
        """
        sources_T = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for mom, coll_dict in self.sources.items():
            for coll, sp_dict in coll_dict.items():
                for species, strata_dict in sp_dict.items():
                    sources_T[mom][species][coll] = strata_dict

        self.sources_by_species = sources_T
        return sources_T

    # -----------------------------------
    # Step 2: regroup by temperature
    # -----------------------------------
    def regroup_by_temperature(self):
        if self.sources_by_species is None:
            self.transpose_sources()

        sources_T = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        temperature = {}

        # Context for mappers (extendable)
        context = self._get_context()

        # -------- Main loop --------
        for mom, sp_dict in self.sources_by_species.items():
            for species, coll_dict in sp_dict.items():
                for coll, strata_dict in coll_dict.items():

                    mapper = self.collision_mappers.get(coll)
                    if mapper is None:
                        continue

                    # Apply collision-specific mapping
                    out = mapper(
                        mom=mom,
                        species=species,
                        strata_dict=strata_dict,
                        temperature_values=self.temperature_values,
                        context=context,
                    )

                    values = out["values"]
                    conservative = out.get("conserved_and_constrained", False)

                    # ---- Conservation check (generic) ----
                    if conservative:
                        input_sum = strata_dict["SUM"]
                        output_sum = sum(values.values())

                        if (not np.allclose(input_sum, output_sum,
                                            rtol=1e-10, atol=1e-12)):
                            raise AssertionError(
                                f"Source not conserved for "
                                f"{mom}/{species}/{coll}"
                            )

                    # ---- Aggregate into temperature bins ----
                    for T_label, arr in values.items():
                        if (
                            T_label is not None
                            and T_label not in self.temperature_values
                        ):
                            T_label = None
                        if sources_T[mom][species][T_label] is None:
                            sources_T[mom][species][T_label] = arr.copy()
                        else:
                            if sources_T[mom][species][T_label].shape != arr.shape:
                                raise ValueError(
                                    f"Shape mismatch for "
                                    f"{mom}/{species}/{T_label}"
                                )
                            sources_T[mom][species][T_label] += arr

                        if T_label is not None:
                            temperature[T_label] = self.temperature_values[T_label]

        self.sources_by_temperature = sources_T
        self.temperature = temperature

        return sources_T, temperature

    def _validate_default_mapper_configuration(self):
        physical_atoms = {
            label.removeprefix("Tn_")
            for label in self.temperature_values
            if label.startswith("Tn_")
            and label != "Tn_ATOMS"
        }
        ion_species = {
            label.removeprefix("Ti_")
            for label in self.temperature_values
            if label.startswith("Ti_")
        }
        if len(physical_atoms) > 1 or len(ion_species) > 1:
            raise ValueError(
                "The built-in CX mapper supports only one atom/ion pair. "
                "Provide collision_mappers explicitly for multi-species "
                "temperature decomposition."
            )

    def _get_context(self):
        context = {
            "particle_sources": {}
        }
        # Pre-extract particle sources (needed for CX)
        if "particle" in self.sources_by_species:
            for species, coll_dict in self.sources_by_species["particle"].items():
                if "atom-plasma" in coll_dict:
                    context["particle_sources"][species] = (
                        coll_dict["atom-plasma"]["SUM"]
                    )
        return context
