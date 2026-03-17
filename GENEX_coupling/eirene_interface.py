import eireneIO
import B2IO
import triangle_mesh

def eirene_interface(eirene_path, b2_path):
    edat = eireneIO.eirene()
    edat.triangle_mesh = triangle_mesh.triangle_mesh(eirene_path)
    b2dat = B2IO.B2(b2_path)

    if (eirene_path / Path("fort.44")).is_file():
        edat.read_ft44(eirene_path / Path("fort.44"))
        ns = edat.fort44["meta"]["npls"]
    else:
        ns = 2

    nx, ny = b2dat.gmtry["vol"].shape

    edat.read_ft31(eirene_path / "fort.31", nx, ny, ns)

    edat.triangle_mesh.calc_incenter()

    return edat, b2dat