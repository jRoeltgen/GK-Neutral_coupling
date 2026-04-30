from netCDF4 import Dataset
import numpy as np

def write_sources_nc(filename, sources, dim_RZ, dim_phi=1,
                     temperature_values=None,
                     write_temperature=False):
    """
    Write sources to NetCDF with structure:

    temperature_<id>/
        attr: temp_id
        temperature (optional)
        <name>/
            <moment>

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
    temps = list(dict.fromkeys(
        t
        for m in sources
        for sp in sources[m]
        for t in sources[m][sp]
    ))

    with Dataset(filename, "w") as nc:

        # global dimensions
        nc.createDimension("dim_RZ", dim_RZ)
        nc.createDimension("dim_phi", dim_phi)

        for i, temp in enumerate(temps):

            # group name
            grp_name = f"temperature_{temp if temp is not None else 'implicit'}"
            temp_grp = nc.createGroup(grp_name)

            # ALWAYS write temp_id attribute
            temp_grp.temperature_id = np.int32(i+1)

            # optional temperature variable
            if write_temperature and temp is not None and temperature_values is not None:

                temp_var = temp_grp.createVariable("temperature", "f8",
                        ("dim_phi", "dim_RZ"))
                val = temperature_values[temp]

                # Normalize shape
                if np.isscalar(val):
                    arr = np.full((dim_phi, dim_RZ), val, dtype=np.float64)

                else:
                    arr = np.asarray(val)

                    if arr.shape == (dim_RZ, dim_phi):
                        arr = arr.T

                    elif arr.shape == (dim_phi, dim_RZ):
                        pass  # already correct

                    else:
                        raise ValueError(
                            f"temperature_values[{temp}] has invalid shape {arr.shape}, "
                            f"expected scalar, ({dim_RZ}, {dim_phi}), or ({dim_phi}, {dim_RZ})"
                        )

                temp_var[:] = arr

            # species groups
            for sp in species:

                sp_grp = temp_grp.createGroup(f"{sp}")

                for mom in moments:

                    temp_dict = sources[mom].get(sp, {})

                    if temp not in temp_dict:
                        continue

                    data = temp_dict[temp]

                    var = sp_grp.createVariable(
                        f"{mom}",
                        "f8",
                        ("dim_phi", "dim_RZ"),
                    )

                    if data.shape == (dim_RZ, dim_phi):
                        var[:] = data.T

                    elif data.shape == (dim_phi, dim_RZ):
                        var[:] = data
                    else:
                        raise ValueError(
                            f"mom[{mom}]/species[{sp}] has invalid shape {data.shape}, "
                            f"expected ({dim_RZ}, {dim_phi}) or ({dim_phi}, {dim_RZ})"
                        )