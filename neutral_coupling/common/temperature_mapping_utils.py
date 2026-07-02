import numpy as np
from scipy.interpolate import NearestNDInterpolator


def default_sparse_temperature_handler(
    temperature,
    density,
    triangle_points,
    *,
    threshold,
):
    """Nearest-fill sparse cells, or omit species that are too sparse."""
    temperature = np.asarray(temperature)
    density = np.asarray(density)
    triangle_points = np.asarray(triangle_points)

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if temperature.shape != density.shape:
        raise ValueError("temperature and density must have the same shape")
    if triangle_points.shape != (density.size, 2):
        raise ValueError(
            "triangle_points must have shape (temperature.size, 2)"
        )

    valid = (density > 0) & np.isfinite(temperature)
    if not np.any(valid) or np.count_nonzero(~valid) / density.size > threshold:
        return None
    if np.all(valid):
        return temperature.copy()

    filled = temperature.copy()
    nearest = NearestNDInterpolator(
        triangle_points[valid], temperature[valid]
    )
    filled[~valid] = nearest(triangle_points[~valid])
    return filled


def default_single_species_pseudo_temperature_handler(
    raw_temperatures,
    density_block,
    *,
    particle_class,
    species_labels,
):
    """Return a pseudo temperature for a group containing exactly one species."""
    if len(raw_temperatures) != 1 or density_block.shape[1] != 1:
        raise ValueError(
            "The default pseudo-species temperature aggregation is only "
            f"valid for one species in particle class '{particle_class}'. "
            f"Found: {species_labels}. Provide a custom "
            "pseudo_temperature_handler for multi-species groups."
        )
    return raw_temperatures[0].copy(), density_block[:, 0].copy()


def resolve_temperature_label(base_label, species, temperature_values):
    """Resolve a species-qualified label, with legacy flat-label fallback."""
    candidates = [f"{base_label}_{species}", base_label]
    if base_label == "Tn":
        candidates.insert(0, f"Tn_{species.rstrip('+')}")
    for label in candidates:
        if label in temperature_values:
            return label
    return None


def split_ionization_cx(St, Sp, Tn, Ti, kB=1.0):
    """
    Split total atom-plasma energy source into ionization + CX components.

    Assumes Maxwellian distributions.
    Assumes only reactions within atom-plasma collisions are ionization & CX
    NOTE: It seems like this should be applicable for all D only simulations
            The change (if any) from other species is TBD

    Parameters
    ----------
    St : array
        total energy source
    Sp : array
        particle source (ionization)
    Tn : float or array
        neutral temperature
    Ti : float or array
        ion temperature
    kB : float
        Boltzmann constant (use 1 if T in energy units)

    Returns
    -------
    Scx : array
        estimated CX event rate
    E_ion : array
        ionization energy contribution
    E_cx : array
        CX energy contribution
    """

    if St.shape != Sp.shape:
        raise ValueError("St and Sp must have the same shape")
    Tn = np.asarray(Tn)
    Ti = np.asarray(Ti)

    pref = 2.0 / (3.0 * kB)

    numerator = pref * St - Sp * Tn
    denom = (Tn - Ti)

    # avoid division blowups
    Scx = np.zeros_like(St)
    if isinstance(denom, (int, float)):
        if denom>1e-12:
            Scx = numerator / denom
    else:
        mask = np.abs(denom) > 1e-12
        Scx[mask] = numerator[mask] / denom[mask]

    # energy components
    E_ion = 1.5 * kB * Sp * Tn
    E_cx  = 1.5 * kB * Scx * (Tn - Ti)

    return Scx, E_ion, E_cx

def linear_temperature_mapper(T_label):
    """
    Maps all moments of a collision linearly to a single temperature.
    """
    def mapper(mom, species, strata_dict, temperature_values, context):
        label = resolve_temperature_label(
            T_label, species, temperature_values
        ) or T_label
        return {
            "values": {label: strata_dict["SUM"]},
            "conserved_and_constrained": True,
        }
    return mapper

def atom_plasma_cx_mapper(
    mom,
    species,
    strata_dict,
    temperature_values,
    context,
):
    """
    Atom–plasma collisions:
    * particle/momentum/etc → neutral temperature
    * energy → split into ionization (Tn) and CX (Ti)
    """
    arr = strata_dict["SUM"]
    neutral_label = resolve_temperature_label(
        "Tn", species, temperature_values
    )
    ion_label = resolve_temperature_label(
        "Ti", species, temperature_values
    )

    # Non-energy moments are linear
    if mom != "energy":
        return {"values":
                    {neutral_label or "Tn": arr},
                "conserved_and_constrained": True,
                }

    if neutral_label is None or ion_label is None:
        return {
            "values": {None: arr},
            "conserved_and_constrained": True,
        }

    # Energy split
    St = arr
    Sp = context["particle_sources"][species]
    Tn = temperature_values[neutral_label]
    Ti = temperature_values[ion_label]

    _, E_ion, E_cx = split_ionization_cx(
        St, Sp, Tn, Ti, kB=1.0
    )

    return {
        "values": {
            neutral_label: E_ion,
            ion_label: E_cx,
        },
        "conserved_and_constrained": False,
    }

default_D_only_collision_mappers = {
    "plasma-plasma":   linear_temperature_mapper("Ti"),
    "molecule-plasma": linear_temperature_mapper("Tm"),
    "testion-plasma":  linear_temperature_mapper("Tti"),
    "photon-plasma":   linear_temperature_mapper("Tph"),
    "atom-plasma":     atom_plasma_cx_mapper,
}
