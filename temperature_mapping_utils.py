import numpy as np

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

    pref = 2.0 / (3.0 * kB)

    numerator = pref * St - Sp * Tn
    denom = (Tn - Ti)

    # avoid division blowups
    Scx = np.zeros_like(St)
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
        return {T_label: strata_dict["SUM"]}
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

    # Non-energy moments are linear
    if mom != "energy":
        return {"Tn": arr}

    # Energy split
    St = arr
    Sp = context["particle_sources"][species]
    Tn = temperature_values["Tn"]
    Ti = temperature_values["Ti"]

    _, E_ion, E_cx = split_ionization_cx(
        St, Sp, Tn, Ti, kB=1.0
    )

    return {
        "Tn": E_ion,
        "Ti": E_cx,
    }

default_D_only_collision_mappers = {
    "plasma-plasma":   linear_temperature_mapper("Ti"),
    "molecule-plasma": linear_temperature_mapper("Tm"),
    "testion-plasma":  linear_temperature_mapper("Tti"),
    "photon-plasma":   linear_temperature_mapper("Tph"),
    "atom-plasma":     atom_plasma_cx_mapper,
}
