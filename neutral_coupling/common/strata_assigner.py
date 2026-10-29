from collections import defaultdict
from . import eirene_input_parser as EireneInputParser
from .extra_fort_schema import PSEUDO_SPECIES
from colorama import Fore, Style
from copy import deepcopy
import numpy as np

class StrataAssigner:
    """
    Assigns sparse EIRENE fort.* output streams to requested strata
    defined in input.dat (or equivalent control file).

    Handles:
      - missing strata (zero-output strata)
      - sparse species emission
      - SUM operator
      - non-Cartesian (stream) output
      - ordered emission

    Data model after assignment:
      sources[moment][collision][stratum_request][species] = array
    """
    default_species = {"atoms"    : ["D", "ATOMS"],
                       "molecules": ["D2", "MOLECULES"],
                       "test_ions": ["D2+", "TEST IONS"],
                       "bulk_ions": ["D+"],
                       "electrons": ["ELECTRONS"]
                       }
    default_strata = "SUM"
    DN_default_vol_rec = {} # See unit test for example

    def __init__(self, requested_strata = default_strata,
                 expected_species_by_particle_class = default_species,
                 vol_rec_mapping = DN_default_vol_rec,
                 inputfile = None, debug = False):
        """
        Parameters
        ----------
        requested_strata : list
            Example: [21, 22, "SUM"]

        expected_species_by_particle_class : dict
            Example:
            {
                "atoms":     ["D", "T"],
                "molecules": ["D2", "DT", "T2"],
                "bulk_ions": ["D+", "T+"],
                "test_ions": ["D2+", "T2+"]
            }
        """
        self.debug = debug
        if inputfile:
            ein = EireneInputParser.EireneInputParser(inputfile).parse_all()
            requested_strata = ein["requested_strata"]
            expected_species_by_particle_class = ein["species"]
            vol_rec_mapping = ein["volume_recombination"]

        self.requested_strata = requested_strata
        self.expected_species_by_particle_class = {
            k: self._upper_set(v) for k, v in expected_species_by_particle_class.items()
        }

        self.volume_recombination = {"particle":{}, "energy":{}}
        self.volume_recombination["particle"] = {
            self._upper_name(k): (set(v) if isinstance(v,
                                        (set, list, tuple)) else {v})
            for k, v in vol_rec_mapping["particle"].items()
        }
        self.volume_recombination["energy"] = {
            self._upper_name(k): (set(v) if isinstance(v,
                                        (set, list, tuple)) else {v})
            for k, v in vol_rec_mapping["energy"].items()
        }

        self._normalize_strata_order()
        #self._build_species_to_class()

        # internal state
        self._group_req_idx = defaultdict(int)

        # output storage
        self.sources = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        self.units = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        self.particle_class = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.contract_part_class = {}

        self.electron_species = self._resolve_single_species("electrons")

    def ingest(self, moment, collision, particle_class, species, units, current_source, debug=False):
        """
        Ingest a single emission record from fort.* stream.

        Parameters
        ----------
        moment : str
        collision : str
        particle_class : str
        species : str
        units : str
        current_source : np.ndarray
        debug : bool (allows debugging a single ingest call)
        """
        # --- Handle "N/A" entries first ---
        if moment == "N/A" or collision == "N/A" or particle_class == "N/A":
            # lazy-init extra_sources dict
            if not hasattr(self, "extra_sources"):
                self.extra_sources = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

            # store the current_source and metadata
            self.extra_sources[moment][collision][species] = {
                "particle_class": particle_class,
                "array": current_source.copy(),
                "units": units
            }
            # skip further ingest for main sources
            return

        group_key = (moment, collision, particle_class, species)

        # group-local strata pointer
        req_idx = self._group_req_idx[group_key]

        allowed = None
        if collision == "plasma-plasma":
            normalized_species = self._upper_name(species)
            raw = self.volume_recombination["particle"].get(
                normalized_species, set()
            )
            raw2 = self.volume_recombination["energy"].get(
                normalized_species, set()
            )
            if isinstance(raw, (int, str)) and isinstance(raw2, (int,str)):
                allowed = {raw, raw2, "SUM"}
            else:
                allowed = set(raw) | {"SUM"} | set(raw2)
        else:
            allowed = set(self.requested_strata)

        # 🔧 skip disallowed strata
        while req_idx < len(self.requested_strata):
            req = self.requested_strata[req_idx]
            if req in allowed:
                break
            req_idx += 1

        self._group_req_idx[group_key] = req_idx

        # exhausted
        if req_idx >= len(self.requested_strata):
            return

        req = self.requested_strata[req_idx]


        if self.debug or debug:
            self._debug_record(moment, collision, particle_class, species, req, allowed)


        # -------------------------
        # plasma-plasma special handling
        # -------------------------
        if collision == "plasma-plasma":
            # advance group-local index to next allowed stratum
            allowed = self._allowed_strata_for_species(species)

            while req_idx < len(self.requested_strata):
                req = self.requested_strata[req_idx]
                if req in allowed:
                    break
                req_idx += 1

            self._group_req_idx[group_key] = req_idx

            if req_idx >= len(self.requested_strata):
                return

            req = self.requested_strata[req_idx]

            if req not in self.sources[moment][collision][species]:
                self.sources[moment][collision][species][req] = np.zeros_like(current_source)

            self.sources[moment][collision][species][req] += current_source
            self.units[moment][collision][species][req] = units
            self.particle_class[moment][collision][species] = particle_class

            self._group_req_idx[group_key] += 1
            return

        # initialize
        if req not in self.sources[moment][collision][species]:
            self.sources[moment][collision][species][req] = np.zeros_like(current_source)

        # shape safety
        if self.sources[moment][collision][species][req].shape != current_source.shape:
            raise ValueError(
                f"Shape mismatch for {moment}/{collision}/{particle_class}/{species} "
                f"stratum {req}: "
                f"{self.sources[moment][collision][species][req].shape} vs {current_source.shape}"
            )

        # accumulate
        self.sources[moment][collision][species][req] += current_source
        self.units[moment][collision][species][req] = units
        self.particle_class[moment][collision][species] = particle_class

        # advance strata for THIS group only
        self._group_req_idx[group_key] += 1

    def sum_over_collisions(self, Te=1.0, Ti=1.0, include_vol_recomb=True):
        """
        Return collision-summed view of sources.

        Rules:
        - Normal strata: sum all collisions
        - SUM strata: skip 'plasma-plasma', optionally include VR
        - Electrons receive additional contributions from bulk ions for particle and energy
        moments (energy scaled by Te/Ti)
        - Raises ValueError if shapes are inconsistent
        - Warnings are printed once per species per stratum for SUM
        (suppressed for momentum and molecules/test_ions/photons)
        """
        total = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        warned_species_strata = set()

        # ------------------------
        # Step 1: sum all collisions
        # ------------------------
        self._sum_collisions(total)

        # ------------------------
        # Step 2: volume recombination (SUM only)
        # ------------------------
        if include_vol_recomb and hasattr(self, "volume_recombination"):
            self._apply_volume_recombination(total, Te, Ti, warned_species_strata)

        return total

    def finalize(self, collisions_to_adjust=None, species_patt_to_ignore="",
                 print_info = False):
        """
        Validation hook after ingestion.
        """
        self._assert_structure()
        self._collapse_incomplete_strata_to_sum(collisions_to_adjust, print_info)
        self._validate_schema(species_patt_to_ignore)

    def _allowed_strata_for_species(self, species):
        allowed = set()
        normalized_species = self._upper_name(species)
        if normalized_species in self.volume_recombination["particle"]:
            allowed.update(
                self.volume_recombination["particle"][normalized_species]
            )
        if normalized_species in self.volume_recombination["energy"]:
            allowed.update(
                self.volume_recombination["energy"][normalized_species]
            )
        allowed.add("SUM")
        return allowed

    # ------------------------
    # Helper: sum all collisions
    # ------------------------
    def _sum_collisions(self, total):
        for mom, coll_dict in self.sources.items():
            for coll, species_dict in coll_dict.items():
                for species, strata_dict in species_dict.items():
                    for stratum, arr in strata_dict.items():
                        if stratum == "SUM" and coll == "plasma-plasma":
                            continue
                        if total[mom][species][stratum] is None:
                            total[mom][species][stratum] = arr.copy()
                        else:
                            if total[mom][species][stratum].shape != arr.shape:
                                raise ValueError(
                                    f"Inconsistent shapes for species {species}, "
                                    f"moment {mom}, stratum {stratum}, collision {coll}: "
                                    f"{total[mom][species][stratum].shape} vs {arr.shape}"
                                )
                            total[mom][species][stratum] += arr


    # ------------------------
    # Helper: volume recombination
    # ------------------------
    def _apply_volume_recombination(self, total, Te, Ti, warned_species_strata):
        applied_vr = set()
        # ------------------------
        # electrons: bulk-ion VR
        # ------------------------
        for mom in {"particle", "energy"}:
            if "plasma-plasma" in self.sources[mom]:
                key = ("electron", mom, self.electron_species)
                if key in applied_vr:
                    raise RuntimeError(
                        f"Electron VR applied twice for {mom}"
                    )
                applied_vr.add(key)
                if mom == "particle":
                    print("")
                self._apply_electron_bulk_vr(
                    mom=mom,
                    coll_dict=self.sources[mom],
                    total=total,
                    vol_attr=None,
                    Te=Te,
                    Ti=Ti,
                    el_species=self.electron_species,
                    warned_species_strata=warned_species_strata,
                    key_warn=("ELECTRONS", mom),
                    suppress_warning=False,
                )

        # ------------------------
        # normal species VR
        # ------------------------
        for mom, coll_dict in self.sources.items():
            if mom=="momentum":
                continue
            vol_attr = self.volume_recombination[mom]
            for species in self.sources[mom].get("plasma-plasma", {}):
                key = ("normal", mom, species)
                if key in applied_vr:
                    raise RuntimeError(
                        f"VR applied twice for {mom}, {species}"
                    )
                applied_vr.add(key)
                # particle class is independent of collision
                particle_cls = next(
                    cls for cls in self.particle_class[mom].values()
                    if species in cls
                )[species]
                suppress_warning = (
                    (mom == "momentum")
                    or (particle_cls in {"molecules", "test_ions", "photons"}))
                key_warn = (species, "SUM")

                if species in vol_attr:
                    self._apply_normal_species_vr(mom, coll_dict, total,
                                                  species, vol_attr,
                                                  warned_species_strata,
                                                  key_warn, suppress_warning)


    # ------------------------
    # Helper: electron bulk-ion VR
    # ------------------------
    def _apply_electron_bulk_vr(self, mom, coll_dict, total, vol_attr, Te, Ti,
                                el_species, warned_species_strata, key_warn, suppress_warning):
        # Get volume recombination
        vr_map = self.volume_recombination[mom]
        # Loop over bulk species and their VR strata sets
        for bulk_species, bulk_strata_set in vr_map.items():
            my_suppress_warning = suppress_warning
            for bulk_stratum in bulk_strata_set:
                included = False
                for c in coll_dict:
                    if c != "plasma-plasma":
                        continue
                    if self.particle_class[mom][c][bulk_species] != "bulk_ions":
                        my_suppress_warning = True
                        continue
                    # bulk_arr is the array for this species, stratum, collision
                    bulk_arr = self.sources[mom][c].get(bulk_species, {}).get(bulk_stratum)
                    if bulk_arr is not None:
                        factor = Te/Ti if mom == "energy" else 1.0
                        # Always add to SUM
                        if total[mom][el_species]["SUM"] is None:
                            total[mom][el_species]["SUM"] = bulk_arr.copy() * factor
                        else:
                            total[mom][el_species]["SUM"] += bulk_arr * factor
                        included = True
                if (not included and key_warn not in warned_species_strata
                    and not my_suppress_warning):
                    print(f"Warning: electron VR from {bulk_species} stratum {bulk_stratum} not found")
                    warned_species_strata.add(key_warn)



    # ------------------------
    # Helper: normal species VR
    # ------------------------
    def _apply_normal_species_vr(self, mom, coll_dict, total, species, vol_attr, warned_species_strata, key_warn, suppress_warning):
        for vol_stratum in vol_attr[species]:
            included = False
            for c in coll_dict:
                if c != "plasma-plasma":
                    continue
                arr = self.sources[mom][c].get(species, {}).get(vol_stratum)
                if arr is not None:
                    if total[mom][species]["SUM"] is None:
                        total[mom][species]["SUM"] = arr.copy()
                    else:
                        total[mom][species]["SUM"] += arr
                    included = True
            if not included and key_warn not in warned_species_strata and not suppress_warning:
                print(f"Warning: volume recombination stratum {vol_stratum} for {species} not found")
                warned_species_strata.add(key_warn)

    def _build_species_to_class(self):
        """
        Build runtime species→particle_class mapping.
        Assumes pseudo-species have already been injected by input parser.
        """
        self.species_to_class = {}

        for cls, species_set in self.expected_species_by_particle_class.items():
            for sp in species_set:
                if sp in self.species_to_class:
                    raise ValueError(
                        f"Species '{sp}' assigned to multiple particle classes: "
                        f"{self.species_to_class[sp]} and {cls}"
                    )
                self.species_to_class[sp] = cls

    def _build_energy_vr_map(self):
        """
        Build VR map in energy-species space (contracted species).
        """
        energy_vr = defaultdict(set)

        for sp, strata in self.volume_recombination.items():
            cls = self.species_to_class.get(sp)

            if cls in PSEUDO_SPECIES:   # atoms, molecules, test_ions
                energy_sp = PSEUDO_SPECIES[cls]
            else:
                energy_sp = sp  # bulk_ions, electrons, etc.

            if isinstance(strata, (set, list, tuple)):
                energy_vr[energy_sp].update(strata)
            else:
                energy_vr[energy_sp].add(strata)

        self.energy_volume_recombination = dict(energy_vr)

    def _collapse_incomplete_strata_to_sum(self, collisions_to_adjust = None,
                                           print_info = True):
        """
        Repair incomplete strata output by EIRENE when zero-valued strata
        are suppressed.

        Assumptions (explicitly enforced):
        1. SUM is always printed by EIRENE.
        2. Printed strata appear in the same order as requested_strata.
        3. If observed_count < expected_count, the *last* observed array
            corresponds to SUM.
        4. Non-SUM strata are not reliable when suppression occurs and
            will be marked as NaN.

        This function:
        - Preserves original sources in self.original_sources
        - Rewrites affected strata dictionaries so that:
            * SUM is correct
            * All other expected strata exist but are NaN
        """

        if not collisions_to_adjust:
            return

        # Preserve original data for debugging / auditing
        self.original_sources = deepcopy(self.sources)

        expected = list(self.requested_strata)

        if "SUM" not in expected:
            raise RuntimeError(
                "Invariant violated: requested_strata does not contain 'SUM'"
            )

        if ("plasma-plasma" in collisions_to_adjust and
            any(self.volume_recombination[type].values())):
            raise RuntimeError(
                "Adjustment of plasma-plasma sources undefined if volume recombination strata requested"
            )

        adjusted = False
        for mom, coll_dict in self.sources.items():
            for coll, species_dict in coll_dict.items():

                if coll not in collisions_to_adjust:
                    continue

                for species, strata_dict in species_dict.items():

                    observed_keys = list(strata_dict.keys())
                    observed_count = len(observed_keys)
                    expected_count = len(expected)

                    if observed_count == expected_count:
                        continue  # complete, nothing to do

                    if observed_count > expected_count:
                        print(
                            f"[WARNING] {mom}/{coll}/{species}: "
                            f"observed more strata ({observed_count}) than expected "
                            f"({expected_count}); skipping adjustment."
                        )
                        continue

                    if observed_count == 0:
                        print(
                            f"[WARNING] {mom}/{coll}/{species}: "
                            f"no strata observed; skipping adjustment."
                        )
                        continue

                    # --- Core assumption: last observed array is SUM ---
                    sum_array = strata_dict[observed_keys[-1]]

                    # Sanity check: shape reference
                    template = sum_array

                    # Build new strata dictionary
                    new_strata = {}

                    for stratum in expected:
                        if stratum == "SUM":
                            new_strata["SUM"] = sum_array
                        else:
                            new_strata[stratum] = np.full_like(template, np.nan)

                    self.sources[mom][coll][species] = new_strata

                    adjusted = True
                    if print_info:
                        print(
                            f"[INFO] {mom}/{coll}/{species}: "
                            f"incomplete strata detected "
                            f"({observed_count}/{expected_count}). "
                            f"Assumed last observed entry is SUM; "
                            f"non-SUM strata set to NaN."
                        )
        if adjusted:
            print(
                f"[WARNING] An incomplete strata was observed.\n"
                f"Assumed last observed entry is SUM and set remaining to NaN.\n"
                f"Original sources saved in self.original_sources.\n"
            )


    def _validate_schema(self, species_patt_to_ignore=""):
        """
        Full structural + physical validation.

        Does NOT enforce numerical SUM equality
        (because partial strata loading is allowed).
        """

        errors = []
        warnings = []

        # ---------------------------------
        # basic structure checks
        # ---------------------------------
        for moment, m_map in self.sources.items():
            for collision, c_map in m_map.items():
                for species, s_map in c_map.items():

                    if not s_map:
                        errors.append(
                            f"Empty emission group: moment={moment}, collision={collision}, species={species}"
                        )

                    # shape consistency within (moment, collision, species)
                    shapes = {arr.shape for arr in s_map.values()}
                    if len(shapes) > 1:
                        errors.append(
                            f"Shape mismatch for moment={moment}, collision={collision}, species={species}: {shapes}"
                        )



        # ---------------------------------
        # strata completeness (per group)
        # ---------------------------------
        expected = set(self.requested_strata)

        for moment, m_map in self.sources.items():
            for collision, c_map in m_map.items():

                # c_map = {species -> {stratum -> array}}
                strata_sets = []
                for species, s_map in c_map.items():
                    strata_sets.append(set(s_map.keys()))

                if not strata_sets:
                    continue

                present = set.union(*strata_sets)   # all strata seen across species
                expected = set(self.requested_strata)

                missing = expected - present
                extra   = present - expected

                if missing:
                    warnings.append(
                        f"Missing strata for {moment}/{collision}: {missing}"
                    )

                if extra:
                    warnings.append(
                        f"Unexpected strata for {moment}/{collision}: {extra}"
                    )

        # ---------------------------------
        # SUM structural consistency
        # ---------------------------------
        for moment, m_map in self.sources.items():
            for species, strata_map in m_map.items():
                if "SUM" not in strata_map:
                    continue

                # SUM must not mix incompatible domains
                # (no numerical check)
                pass

        # ---------------------------------
        # particle-class domain consistency
        # ---------------------------------
        for moment, m_map in self.sources.items():
            for collision, c_map in m_map.items():
                for species in c_map.keys():
                    if self._contains_any(species_patt_to_ignore, species):
                        continue
                    normalized_species = self._upper_name(species)
                    found = False
                    for domain in self.expected_species_by_particle_class.values():
                        if normalized_species in domain:
                            found = True
                            break
                    if not found:
                        warnings.append(
                            f"Species '{species}' not in any particle-class domain"
                        )



        # ---------------------------------
        # volume recombination consistency
        # ---------------------------------
        for key in self.volume_recombination.keys():
            for sp, vr in self.volume_recombination[key].items():
                found = False
                for m in self.sources:
                    for c in self.sources[m]:
                        if any(
                            self._upper_name(source_species) == sp
                            for source_species in self.sources[m][c]
                        ):
                            found = True
                            break

                if not found:
                    warnings.append(
                        f"Volume recombination ({key}) species '{sp}' not present in sources"
                    )

        # ----------------------
        # print debug summary if there are any warnings/errors
        # ----------------------
        if errors or warnings:
            print("\n[DEBUG SUMMARY OF INPUT EXPECTATIONS]")
            print("Requested strata:", self.requested_strata)
            print("Expected species per particle class:")
            for pc, domain in self.expected_species_by_particle_class.items():
                print(f"  {pc}: {sorted(domain)}")
            print("Volume recombination mapping (particle):")
            for sp, stratum in self.volume_recombination["particle"].items():
                print(f"  {sp} -> {stratum}")
            print("Volume recombination mapping (energy):")
            for sp, stratum in self.volume_recombination["energy"].items():
                print(f"  {sp} -> {stratum}")
            print()

        # ---------------------------------
        # reporting
        # ---------------------------------
        if errors:
            raise RuntimeError(
                "Strata schema validation failed:\n" + "\n".join(errors)
            )

        if warnings:
            print("\n[Strata schema warnings]")
            for w in warnings:
                print("  -", w)

    def __advance_request(self):
        """Move to next requested stratum operator."""
        self._req_idx += 1
        if self._req_idx < len(self.requested_strata):
            self._current_request = self.requested_strata[self._req_idx]
            self._seen_species = set()
        else:
            self._current_request = None

    def _normalize_strata_order(self):
        nums = sorted(s for s in self.requested_strata if s != "SUM")
        if "SUM" in self.requested_strata:
            nums.append("SUM")
        self.requested_strata = nums

    def _ensure_array(self, slot, template):
        if slot is None:
            return np.zeros_like(template)
        if not hasattr(slot, "shape"):
            return np.zeros_like(template) + slot
        return slot

    def _upper_name(self, x):
        return x.upper() if isinstance(x, str) else x

    def _upper_set(self, v):
        if isinstance(v, (set, list, tuple)):
            return {self._upper_name(x) for x in v}
        return {self._upper_name(v)}

    def _assert_structure(self):
        """
        Structural invariant check for sources layout.

        Enforces:
        sources[moment][collision][species][stratum] = np.ndarray
        """
        for moment, m_map in self.sources.items():
            if not isinstance(m_map, dict):
                raise RuntimeError(f"Invalid structure at sources[{moment}]")

            for collision, c_map in m_map.items():
                if not isinstance(c_map, dict):
                    raise RuntimeError(
                        f"Invalid structure at sources[{moment}][{collision}]"
                    )

                for species, s_map in c_map.items():
                    if not isinstance(s_map, dict):
                        raise RuntimeError(
                            f"Invalid structure at sources[{moment}][{collision}][{species}]"
                        )

                    for stratum, arr in s_map.items():
                        if not isinstance(arr, np.ndarray):
                            raise RuntimeError(
                                f"Invalid leaf at "
                                f"sources[{moment}][{collision}][{species}][{stratum}] "
                                f"(type={type(arr)})"
                            )

    def _debug_record(self, moment, collision, particle_class, species, stratum, allowed_strata):

        # -------- species validation --------
        species_expected = any(
            self._upper_name(species) in domain
            for domain in self.expected_species_by_particle_class.values()
        )
        species_color = Fore.GREEN if species_expected else Fore.RED

        # -------- stratum validation --------
        stratum_expected = stratum in allowed_strata
        stratum_color = Fore.GREEN if stratum_expected else Fore.RED

        # -------- mismatches --------
        mismatches = []
        if not species_expected:
            mismatches.append("species")
        if not stratum_expected:
            mismatches.append("stratum")

        # -------- output --------
        print(
            f"[DEBUG] moment={moment}, collision={collision}, class={particle_class}, "
            f"species={species_color}{species}{Style.RESET_ALL}, "
            f"stratum={stratum_color}{stratum}{Style.RESET_ALL}, "
            f"allowed_strata={sorted(allowed_strata, key=lambda x: str(x))}"
            + (f" | unexpected: {', '.join(mismatches)}" if mismatches else "")
        )

    def _resolve_single_species(self, cls: str) -> str:
        species = self.expected_species_by_particle_class.get(cls, [])
        if len(species) != 1:
            raise ValueError(
                f"Expected exactly one species for particle class '{cls}', "
                f"found {species}"
            )
        return next(iter(species))

    def _contains_any(self, x, s):
        if isinstance(x, str):
            candidates = [x]
        elif isinstance(x, (list, tuple)):
            candidates = x
        else:
            raise TypeError("x must be a string or list of strings")

        return any(sub in s for sub in candidates)
