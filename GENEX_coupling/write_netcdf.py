from netCDF4 import Dataset

def write_sources_nc(filename, sources, temperature_values=None,
                     write_temperature=False):
    """
    Write sources to NetCDF with structure:

    temperature_<id>/
        attr: temp_id
        temperature (optional)
        species_<name>/
            mom_<moment>

    Parameters
    ----------
    filename : str
        Output NetCDF filename

    sources : dict
        sources[moment][species][temperature] -> ndarray

    temperature_values : dict, optional
        temperature_values[temp_key] -> temperature value/array

    write_temperature : bool
        Whether to write the temperature variable
    """

    moments = list(sources.keys())
    species = list(next(iter(sources.values())).keys())

    # collect temperature keys
    temps = {
        t
        for m in sources
        for sp in sources[m]
        for t in sources[m][sp]
    }

    implicit = None in temps
    temps.discard(None)

    temps = sorted(temps)
    if implicit:
        temps = [None] + temps

    with Dataset(filename, "w") as nc:

        # global dimensions
        nc.createDimension("RZ", None)
        nc.createDimension("phi", None)

        for temp in temps:

            # group name
            grp_name = f"temperature_{temp if temp is not None else 'implicit'}"
            temp_grp = nc.createGroup(grp_name)

            # ALWAYS write temp_id attribute
            temp_grp.temp_id = "implicit" if temp is None else str(temp)

            # optional temperature variable
            if write_temperature and temp is not None and temperature_values is not None:

                temp_var = temp_grp.createVariable("temperature", "f8")
                temp_var[:] = temperature_values[temp]

            # species groups
            for sp in species:

                sp_grp = temp_grp.createGroup(f"species_{sp}")

                for mom in moments:

                    temp_dict = sources[mom].get(sp, {})

                    if temp not in temp_dict:
                        continue

                    data = temp_dict[temp]

                    var = sp_grp.createVariable(
                        f"mom_{mom}",
                        "f8",
                        ("RZ", "phi"),
                    )

                    var[:] = data