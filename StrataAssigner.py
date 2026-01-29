from collections import defaultdict
import EireneInputParser
import colorama
from colorama import Fore, Style
import numpy as np
import pdb

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
    default_species = {"atoms"    : ["D"],
                       "molecules": ["D2"],
                       "test_ions": ["D2+"],
                       "bulk_ions": ["D+"],
                       "electrons": ["ELECTRONS"]
                       }
    default_strata = "SUM"
    DN_default_vol_rec = {"D+"        : 11,
                          "D"         : 11
                         }

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
            k: set(v) for k, v in expected_species_by_particle_class.items()
        }
        self.volume_recombination = {
            k: (set(v) if isinstance(v, (set, list, tuple)) else {v})
            for k, v in vol_rec_mapping.items()
        }
        self._normalize_strata_order()

        # internal state
        self._group_req_idx = defaultdict(int)

        # output storage
        self.sources = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        self.units = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        self.particle_class = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def ingest(self, moment, collision, particle_class, species, units, current_source):
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
            raw = self.volume_recombination.get(species, set())
            if isinstance(raw, (int, str)):
                allowed = {raw, "SUM"}
            else:
                allowed = set(raw) | {"SUM"}
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


        if self.debug:
            self._debug_record(moment, collision, particle_class, species, req, allowed)


        # -------------------------
        # plasma-plasma special handling
        # -------------------------
        if collision == "plasma-plasma":

            # allowed strata for this species
            allowed = set()

            # volume recombination stratum
            if species in self.volume_recombination:
                allowed.update(self.volume_recombination[species])

            # SUM is always allowed
            allowed.add("SUM")

            # assign in file-order (lowest → highest → SUM)
            for req in self.requested_strata:
                if req not in allowed:
                    continue

                if req not in self.sources[moment][collision][species]:
                    self.sources[moment][collision][species][req] = np.zeros_like(current_source)

                if self.sources[moment][collision][species][req].shape != current_source.shape:
                    raise ValueError(
                        f"Shape mismatch for {moment}/{collision}/{particle_class}/{species} "
                        f"stratum {req}: "
                        f"{self.sources[moment][collision][species][req].shape} vs {current_source.shape}"
                    )

                self.sources[moment][collision][species][req] += current_source
                self.units[moment][collision][species][req] = units
                self.particle_class[moment][collision][species] = particle_class

            return   # do not fall through

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

    def finalize(self):
        """
        Validation hook after ingestion.
        """
        self._assert_structure()
        self._validate_schema()

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
        vol_attr = self.volume_recombination
        for mom, coll_dict in self.sources.items():
            for coll, species_dict in coll_dict.items():
                for species in species_dict:
                    # Only SUM gets VR
                    if "SUM" not in total[mom][species]:
                        continue

                    particle_cls = self.particle_class[mom][coll][species]
                    suppress_warning = (mom == "momentum") or (particle_cls in {"molecules", "test_ions", "photons"})
                    key_warn = (species, "SUM")

                    # ------------------------
                    # electrons: bulk-ion VR
                    # ------------------------
                    if particle_cls == "ELECTRONS" and mom in {"particle", "energy"}:
                        self._apply_electron_bulk_vr(mom, coll_dict, total, vol_attr, Te, Ti, warned_species_strata, key_warn, suppress_warning)

                    # ------------------------
                    # normal species VR
                    # ------------------------
                    if species in vol_attr:
                        print(vol_attr)
                        self._apply_normal_species_vr(mom, coll_dict, total, species, vol_attr, warned_species_strata, key_warn, suppress_warning)


    # ------------------------
    # Helper: electron bulk-ion VR
    # ------------------------
    def _apply_electron_bulk_vr(self, mom, coll_dict, total, vol_attr, Te, Ti, warned_species_strata, key_warn, suppress_warning):
        # Loop over bulk species and their VR strata sets
        for bulk_species, bulk_strata_set in vol_attr.items():
            for bulk_stratum in bulk_strata_set:
                included = False
                for c in coll_dict:
                    if c != "plasma-plasma":
                        continue
                    # bulk_arr is the array for this species, stratum, collision
                    bulk_arr = self.sources[mom][c].get(bulk_species, {}).get(bulk_stratum)
                    if bulk_arr is not None:
                        factor = Te/Ti if mom == "energy" else 1.0
                        # Always add to SUM
                        if total[mom]["e-"]["SUM"] is None:
                            total[mom]["e-"]["SUM"] = bulk_arr.copy() * factor
                        else:
                            total[mom]["e-"]["SUM"] += bulk_arr * factor
                        included = True
                if not included and key_warn not in warned_species_strata and not suppress_warning:
                    print(f"Warning: electron VR from {bulk_species} stratum {bulk_stratum} not found")
                    warned_species_strata.add(key_warn)



    # ------------------------
    # Helper: normal species VR
    # ------------------------
    def _apply_normal_species_vr(self, mom, coll_dict, total, species, vol_attr, warned_species_strata, key_warn, suppress_warning):
        for vol_stratum in vol_attr[species]:
            included = False
            for c in coll_dict:
                if c == "plasma-plasma":
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


    def _validate_schema(self):
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
                    found = False
                    for domain in self.expected_species_by_particle_class.values():
                        if species in domain:
                            found = True
                            break
                    if not found:
                        warnings.append(
                            f"Species '{species}' not in any particle-class domain"
                        )



        # ---------------------------------
        # volume recombination consistency
        # ---------------------------------
        for sp, vr in self.volume_recombination.items():
            found = False
            for m in self.sources:
                for c in self.sources[m]:
                    if sp in self.sources[m][c]:
                        found = True
                        break

            if not found:
                warnings.append(
                    f"Volume recombination species '{sp}' not present in sources"
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
            print("Volume recombination mapping:")
            for sp, stratum in self.volume_recombination.items():
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
        if not getattr(self, "debug", False):
            return

        # -------- species validation --------
        species_expected = any(
            species in domain for domain in self.expected_species_by_particle_class.values()
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
