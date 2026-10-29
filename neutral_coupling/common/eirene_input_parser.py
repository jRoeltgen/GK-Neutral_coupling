import re
import math
from pathlib import Path
from .extra_fort_schema import PSEUDO_SPECIES
import scipy.constants as pyconst

class EireneInputParser:
    """
    Parser for EIRENE input.dat-like files.

    Extracts:
      - requested strata (SUM, 1..N)
      - species lists by particle class
      - unified volume recombination mapping {species -> stratum}
        (atomic species + corresponding bulk ion species)
    """

    PARTICLE_HEADERS = {
        "atoms": "** 4a. Neutral atom species",
        "molecules": "** 4b. Neutral molecule species",
        "test_ions": "** 4c. Test ion species",
        "bulk_ions": "** 5a. Bulk ion species",
    }

    def __init__(self, input_file: str | Path):
        self.input_file = Path(input_file)
        self.lines = self.input_file.read_text().splitlines()
        self.report = ValidationReport()
        self.requested_strata = []          # e.g. ["SUM", 21, 22]
        self.species = {                    # particle class -> [species]
            "atoms": [],
            "molecules": [],
            "test_ions": [],
            "bulk_ions": [],
            "electrons": ["ELECTRONS"]
        }
        self.masses = {                    # particle class -> [species]
            "atoms": [],
            "molecules": [],
            "test_ions": [],
            "bulk_ions": [],
            "electrons": [pyconst.electron_mass]
        }

        # unified mapping: atomic + bulk species -> stratum
        self.volume_recombination = {"particle":{}, "energy":{}} # {species_name: stratum}

    # ---------------------------
    # MASTER
    # ---------------------------

    def parse_all(self):
        self.parse_requested_strata()
        self.parse_species()
        # add EIRENE internal sum-species
        self.add_internal_group_species()
        self.parse_volume_recombination()
        self.propagate_volume_recombination_to_pseudo_species()

        self.validate_schema()
        if self.report.has_issues():
            self.report.print()
            print("printing debug summary...")
            self._print_debug_summary()

        return {
            "requested_strata": self.requested_strata,
            "species": self.species,
            "volume_recombination": self.volume_recombination,
            "validation": self.report
        }

    # ---------------------------
    # STRATA REQUEST PARSING
    # ---------------------------

    def parse_requested_strata(self):
        for i, line in enumerate(self.lines):
            if "*** 11. Data for numerical and graphical output" in line:
                tf_line = self.lines[i+2].strip()
                tf = tf_line.replace(" ", "")

                # order: SUM,1,2,...
                for idx, ch in enumerate(tf):
                    if ch == 'T':
                        if idx == 0:
                            self.requested_strata.append("SUM")
                        else:
                            self.requested_strata.append(idx)

                # possible continuation line
                if i+3 < len(self.lines) and re.fullmatch(r"[TF ]+", self.lines[i+3].strip()):
                    tf2 = self.lines[i+3].replace(" ", "")
                    offset = len(tf)
                    for j, ch in enumerate(tf2):
                        if ch == 'T':
                            self.requested_strata.append(offset + j)
                break

    # ---------------------------
    # SPECIES PARSING
    # ---------------------------
    def _parse_species_block(self, start_idx):
        i = start_idx + 1

        # number of species
        n = int(self.lines[i].strip())
        i += 1
        species = []
        mass = []
        for _ in range(n):
            # advance until we hit a real species-definition line
            while i < len(self.lines) and not self._is_species_line(self.lines[i]):
                i += 1
            line = self.lines[i]
            tokens = line.split()
            # extract species name = first non-numeric token after index
            name = None
            for tok in tokens[1:]:
                if not re.fullmatch(r"-?\d+(\.\d+)?([Ee][+-]?\d+)?", tok):
                    name = tok
                    break
            if name is None:
                raise ValueError(f"Failed to detect species name in line: {line}")

            # extract all integers
            ints = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(ints) < 10:
                raise ValueError(
                    f"Cannot parse stride for species '{name}'. "
                    f"Expected at least 10 integers in species line:\n{line}"
                )

            stride = ints[9]   # true 10th integer
            species.append(name)
            # The mass number immediately follows the species name. Reading
            # it by token position avoids treating digits in names such as
            # D2, T2+, or D2T2(B) as numeric fields.
            name_index = tokens.index(name)
            mass_number = int(tokens[name_index + 1])
            mass.append(mass_number * pyconst.proton_mass)
            # jump over irrelevant lines
            i += 1 + 2 * stride

        return species, mass

    def _is_species_line(self, line: str) -> bool:
        tokens = line.split()
        if len(tokens) < 2:
            return False
        # must contain a non-numeric species token
        return any(not re.fullmatch(r"-?\d+(\.\d+)?([Ee][+-]?\d+)?", t) for t in tokens[1:])


    def parse_species(self):
        for i, line in enumerate(self.lines):
            for cls, header in self.PARTICLE_HEADERS.items():
                if line.strip().startswith(header):
                    sp, mass = self._parse_species_block(i)
                    if cls == "bulk_ions":
                        # remove (B) species
                        physical = [
                            (species, species_mass)
                            for species, species_mass in zip(sp, mass)
                            if "(B)" not in species
                        ]
                        sp = [species for species, _ in physical]
                        mass = [species_mass for _, species_mass in physical]
                    self.species[cls].extend(sp)
                    self.masses[cls].extend(mass)

    # ---------------------------
    # VOLUME RECOMBINATION
    # ---------------------------

    def parse_volume_recombination(self):
        # build bulk-ion index -> species name map
        bulk_index_map = {}
        for idx, sp in enumerate(self.species.get("bulk_ions", []), start=1):
            bulk_index_map[idx] = sp

        for i, line in enumerate(self.lines):
            if line.strip().startswith("*") and "Volumetric recombination" in line:
                # example: *  22 : Volumetric recombination T
                m = re.search(r"\*\s*(\d+)\s*:\s*Volumetric recombination\s+(\S+)", line)
                if not m:
                    continue

                stratum = int(m.group(1))
                atomic_species = m.group(2)

                # add atomic species
                self.volume_recombination["particle"][atomic_species] = stratum

                # 5 lines below → bulk species index
                bulk_line = self.lines[i+5].strip()
                try:
                    bulk_index = int(bulk_line)
                except ValueError:
                    continue

                bulk_species = bulk_index_map.get(bulk_index, None)
                if bulk_species is not None:
                    # add bulk ion species with same stratum
                    self.volume_recombination["particle"][bulk_species] = stratum
                else:
                    print(
                        f"Warning: bulk ion index {bulk_index} not found for "
                        f"volumetric recombination of {atomic_species}"
                    )

    def add_internal_group_species(self):
        """
        Add EIRENE internal summed species groups.

        EIRENE internally provides:
        - ATOMS       = sum over atomic species
        - TEST IONS   = sum over test ion species
        - MOLECULES   = sum over molecular species
        """
        energy_contract_sp = ["atoms", "test_ions", "molecules"]
        for key in energy_contract_sp:
            if len(self.species.get(key, [])) >= 1:
                if PSEUDO_SPECIES[key] not in self.species[key]:
                    self.species[key].append(PSEUDO_SPECIES[key])

    def propagate_volume_recombination_to_pseudo_species(self):
        """
        Map physical VR strata onto EIRENE pseudo-species
        (ATOMS, MOLECULES, TEST IONS) for energy channels.
        """
        if not hasattr(self, "volume_recombination"):
            return
        vr = self.volume_recombination["particle"]
        new_vr = {}

        for cls, pseudo in PSEUDO_SPECIES.items():
            # physical species in this particle class
            phys_species = self.species.get(cls, [])
            if not phys_species:
                continue

            # collect VR strata from physical species
            strata = set()
            for sp in phys_species:
                if sp in vr:
                    s = vr[sp]
                    if isinstance(s, (set, list, tuple)):
                        strata |= set(s)
                    else:
                        strata.add(s)

            # assign to pseudo-species
            if strata:
                new_vr[pseudo] = strata
        self.volume_recombination["energy"] = new_vr


    # ---------------------------
    # CONSISTENCY VALIDATION
    # ---------------------------

    def validate_schema(self):
        # strata
        if not self.requested_strata:
            self.report.error("No strata were requested in input.dat")


        # species presence
        for cls, sp_list in self.species.items():
            if not sp_list:
                self.report.warn(f"No species found for particle class '{cls}'")

        # Every physical species must have exactly one finite, positive mass.
        pseudo_species = set(PSEUDO_SPECIES.values())
        for cls, sp_list in self.species.items():
            physical_species = [sp for sp in sp_list if sp not in pseudo_species]
            mass_list = self.masses.get(cls)

            if mass_list is None:
                self.report.error(
                    f"No masses found for particle class '{cls}'"
                )
                continue

            if len(mass_list) != len(physical_species):
                self.report.error(
                    f"Mass count mismatch for particle class '{cls}': "
                    f"{len(physical_species)} physical species, "
                    f"{len(mass_list)} masses"
                )

            for index, mass in enumerate(mass_list):
                species = (
                    physical_species[index]
                    if index < len(physical_species)
                    else f"mass index {index}"
                )
                if (
                    not isinstance(mass, (int, float))
                    or not math.isfinite(mass)
                    or mass <= 0
                ):
                    self.report.error(
                        f"Invalid mass for '{species}' in particle class "
                        f"'{cls}': {mass!r}"
                    )

        # volume recombination consistency
        if "SUM" in self.requested_strata:
            for _,type in self.volume_recombination.items():
                for atom, strata in type.items():
                    if atom not in self.species.get("atoms", []) and atom not in self.species.get("bulk_ions", []):
                        self.report.warn(
                            f"Volume recombination defined for '{atom}', but species not found in atom list"
                        )


                    # normalize strata to a set
                    if isinstance(strata, (set, list, tuple)):
                        strata_set = set(strata)
                    else:
                        strata_set = {strata}

                    missing = strata_set - set(self.requested_strata)

                    if missing:
                        self.report.warn(
                            f"Volume recombination strata {sorted(missing)} for '{atom}' not in requested strata list"
                        )


    def _print_debug_summary(self):
        print("\n[EIRENE INPUT DEBUG SUMMARY]")
        print("\nRequested strata:")
        print(" ", self.requested_strata)

        print("\nSpecies by particle class:")
        for cls, sp in self.species.items():
            print(f"  {cls}: {sp}")

        print("\nMasses by particle class:")
        for cls, masses in self.masses.items():
            print(f"  {cls}: {masses}")

        print("\nVolume recombination mapping:")
        if self.volume_recombination:
            for sp, st in self.volume_recombination.items():
                print(f"  {sp} -> {st}")
        else:
            print("  (none)")



class ValidationReport:
    def __init__(self):
        self.warnings = []
        self.errors = []


    def warn(self, msg):
        self.warnings.append(msg)


    def error(self, msg):
        self.errors.append(msg)


    def has_issues(self):
        return bool(self.warnings or self.errors)


    def print(self):
        if self.errors:
            print("\n[EIRENE INPUT VALIDATION ERRORS]")
            for e in self.errors:
                print(" -", e)


        if self.warnings:
            print("\n[EIRENE INPUT VALIDATION WARNINGS]")
            for w in self.warnings:
                print(" -", w)
