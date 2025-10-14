import numpy as np
import warnings
import triangle_mesh as triangles
import scipy.constants as pyconst

class B2:
    def __init__(self):
        self.state = {}
        self.gmtry = {}
        
    def read_b2fstate(self, filename="./b2fstate"):
        """
        Read b2fstati/b2fstate file created by B2.5.
        
        Output is a dictionary "state" with all the data fields in the b2fstate/i
        file 
        """
        
        # Open file
        try:
            fid = open(filename, "r")
        except OSError as e:
            raise RuntimeError(f"Could not open file: {e}")
        
        # Get version of the b2fstate file
        line = fid.readline()
        version = line[7:17]   # MATLAB indexing (8:17) -> Python slice (7:17)
        print(f"read_b2fstate -- file version {version}")
        self.state["version"] = version
        
        # Read dimensions nx, ny, ns
        dim = self.__read_field("int",fid, "nx,ny,ns", [3])
        nx, ny, ns = dim
        self.state["dim"] = [nx+2, ny+2, ns]
        
        fluxdim  = [nx+2, ny+2, 2]
        fluxdimp = [nx+2, ny+2]
        fluxdims = [nx+2, ny+2, 2, ns]
        
        if version >= "03.001.000":
            fluxdim  = [nx+2, ny+2, 2, 2]
            fluxdimp = fluxdim
            fluxdims = [nx+2, ny+2, 2, 2, ns]
            
        # Read charges etc.
        self.state["zamin"] = self.__read_field("float", fid, "zamin", [ns])
        self.state["zamax"] = self.__read_field("float", fid, "zamax", [ns])
        self.state["zn"]    = self.__read_field("float", fid, "zn   ", [ns])
        self.state["am"]    = self.__read_field("float", fid, "am   ", [ns])
        
        # Read state variables
        self.state["na"]    = self.__read_field("float", fid, "na",     [nx+2, ny+2, ns])
        self.state["ne"]    = self.__read_field("float", fid, "ne",     [nx+2, ny+2])
        self.state["ua"]    = self.__read_field("float", fid, "ua",     [nx+2, ny+2, ns])
        self.state["uadia"] = self.__read_field("float", fid, "uadia",  [nx+2, ny+2, 2, ns])
        self.state["te"]    = self.__read_field("float", fid, "te",     [nx+2, ny+2])
        self.state["ti"]    = self.__read_field("float", fid, "ti",     [nx+2, ny+2])
        self.state["po"]    = self.__read_field("float", fid, "po",     [nx+2, ny+2])
            
        # Read fluxes
        self.state["fna"]     = self.__read_field("float", fid, "fna",    fluxdims)
        self.state["fhe"]     = self.__read_field("float", fid, "fhe",    fluxdim)
        self.state["fhi"]     = self.__read_field("float", fid, "fhi",    fluxdim)
        self.state["fch"]     = self.__read_field("float", fid, "fch",    fluxdim)
        self.state["fch_32"]  = self.__read_field("float", fid, "fch_32", fluxdim)
        self.state["fch_52"]  = self.__read_field("float", fid, "fch_52", fluxdim)
        self.state["kinrgy"]  = self.__read_field("float", fid, "kinrgy", [nx+2, ny+2, ns])
        self.state["time"]    = self.__read_field("float", fid, "time",   [1])
        self.state["fch_p"]   = self.__read_field("float", fid, "fch_p",  fluxdimp)
        
        # Compare versions numerically (strip dots for consistency with MATLAB)
        def version_to_num(v):
            return int(v.replace(".", ""))
        
        if version_to_num(version) >= version_to_num("03.000.005"):
            # Extra fields from version 03.000.005 onwards
            self.state["fna_mdf"]     = self.__read_field("float", fid, "fna_mdf",    fluxdims)
            self.state["fhe_mdf"]     = self.__read_field("float", fid, "fhe_mdf",    fluxdim)
            self.state["fhi_mdf"]     = self.__read_field("float", fid, "fhi_mdf",    fluxdim)
            self.state["fna_fcor"]    = self.__read_field("float", fid, "fna_fcor",   fluxdims)
            self.state["fna_nodrift"] = self.__read_field("float", fid, "fna_nodrift",fluxdims)
            self.state["fna_he"]      = self.__read_field("float", fid, "fna_he",     fluxdims)
            self.state["fnaPSch"]     = self.__read_field("float", fid, "fnaPSch",    fluxdims)
            self.state["fhePSch"]     = self.__read_field("float", fid, "fhePSch",    fluxdim)
            self.state["fhiPSch"]     = self.__read_field("float", fid, "fhiPSch",    fluxdim)
            self.state["fna_eir"]     = self.__read_field("float", fid, "fna_eir",    fluxdims)
            self.state["fne_eir"]     = self.__read_field("float", fid, "fne_eir",    fluxdim)
            self.state["fhe_eir"]     = self.__read_field("float", fid, "fhe_eir",    fluxdim)
            self.state["fhi_eir"]     = self.__read_field("float", fid, "fhi_eir",    fluxdim)
            self.state["fna_32"]      = self.__read_field("float", fid, "fna_32",     fluxdims)
            self.state["fna_52"]      = self.__read_field("float", fid, "fna_52",     fluxdims)
            self.state["fni_32"]      = self.__read_field("float", fid, "fni_32",     fluxdim)
            self.state["fni_52"]      = self.__read_field("float", fid, "fni_52",     fluxdim)
            self.state["fne_32"]      = self.__read_field("float", fid, "fne_32",     fluxdim)
            self.state["fne_52"]      = self.__read_field("float", fid, "fne_52",     fluxdim)
            self.state["fchdia"]      = self.__read_field("float", fid, "fchdia",     fluxdim)
            self.state["fchin"]       = self.__read_field("float", fid, "fchin",      fluxdim)
            self.state["fchvispar"]   = self.__read_field("float", fid, "fchvispar",  fluxdim)
            self.state["fchvisper"]   = self.__read_field("float", fid, "fchvisper",  fluxdim)
            self.state["fchvisq"]     = self.__read_field("float", fid, "fchvisq",    fluxdim)
            self.state["fchinert"]    = self.__read_field("float", fid, "fchinert",   fluxdim)
            
            self.state["vaecrb"] = self.__read_field("float", fid, "vaecrb", [nx+2, ny+2, 2, ns])
            self.state["vadia"]  = self.__read_field("float", fid, "vadia",  [nx+2, ny+2, 2, ns])
            self.state["wadia"]  = self.__read_field("float", fid, "wadia",  [nx+2, ny+2, 2, ns])
            self.state["veecrb"] = self.__read_field("float", fid, "veecrb", [nx+2, ny+2, 2])
            self.state["vedia"]  = self.__read_field("float", fid, "vedia",  [nx+2, ny+2, 2])
            
            self.state["floe_noc"] = self.__read_field("float", fid, "floe_noc", fluxdim)
            self.state["floi_noc"] = self.__read_field("float", fid, "floi_noc", fluxdim)
            
        # Close file
        fid.close()

    def read_b2fgmtry(self, filename="./b2fgmtry"):
        """
        Read b2fgmtry file created by B2.5.

        Output is a SimpleNamespace 'gmtry' with all the data fields
        in the b2fgmtry file.
        """

        fields = {}

        # Open file
        try:
            fid = open(filename, "r")
        except OSError as e:
            raise RuntimeError(f"Could not open file: {e}")
        
        # Get version of the b2fgmtry file
        line = fid.readline()
        version = line[7:17]   # MATLAB (8:17) -> Python slice (7:17)
        fields["version"] = version
        print(f"read_b2fgmtry -- file version {version}")
        
        # Read dimensions nx, ny
        dim = self.__read_field("int",fid, "nx,ny", [2])
        nx, ny = dim
        
        # Expected array sizes
        qcdim = [nx+2, ny+2]
        if version >= "03.001.000":
            qcdim = [nx+2, ny+2, 2]
            
        # --- Symmetry ---
        fields["isymm"] = self.__read_field("int", fid, "isymm", [1])
            
        # --- Geometry variables ---
        fields["crx"]  = self.__read_field("float", fid, "crx",  [nx+2, ny+2, 4])
        fields["cry"]  = self.__read_field("float", fid, "cry",  [nx+2, ny+2, 4])
        fields["fpsi"] = self.__read_field("float", fid, "fpsi", [nx+2, ny+2, 4])
        fields["ffbz"] = self.__read_field("float", fid, "ffbz", [nx+2, ny+2, 4])
        fields["bb"]   = self.__read_field("float", fid, "bb",   [nx+2, ny+2, 4])
        fields["vol"]  = self.__read_field("float", fid, "vol",  [nx+2, ny+2])
        fields["hx"]   = self.__read_field("float", fid, "hx",   [nx+2, ny+2])
        fields["hy"]   = self.__read_field("float", fid, "hy",   [nx+2, ny+2])
        fields["qz"]   = self.__read_field("float", fid, "qz",   [nx+2, ny+2, 2])
        fields["qc"]   = self.__read_field("float", fid, "qc",   qcdim)
        fields["gs"]   = self.__read_field("float", fid, "gs",   [nx+2, ny+2, 3])
        
        # --- Other geometrical parameters ---
        fields["nlreg"] = self.__read_field("int", fid, "nlreg", [1])
        fields["nlxlo"] = self.__read_field("int", fid, "nlxlo", fields["nlreg"])
        fields["nlxhi"] = self.__read_field("int", fid, "nlxhi", fields["nlreg"])
        fields["nlylo"] = self.__read_field("int", fid, "nlylo", fields["nlreg"])
        fields["nlyhi"] = self.__read_field("int", fid, "nlyhi", fields["nlreg"])
        fields["nlloc"] = self.__read_field("int", fid, "nlloc", fields["nlreg"])
        
        fields["nncut"]     = self.__read_field("int", fid, "nncut", [1])
        fields["leftcut"]   = self.__read_field("int", fid, "leftcut",   fields["nncut"])
        fields["rightcut"]  = self.__read_field("int", fid, "rightcut",  fields["nncut"])
        fields["topcut"]    = self.__read_field("int", fid, "topcut",    fields["nncut"])
        fields["bottomcut"] = self.__read_field("int", fid, "bottomcut", fields["nncut"])
        
        fields["leftix"]   = self.__read_field("int", fid, "leftix",   [nx+2, ny+2])
        fields["rightix"]  = self.__read_field("int", fid, "rightix",  [nx+2, ny+2])
        fields["topix"]    = self.__read_field("int", fid, "topix",    [nx+2, ny+2])
        fields["bottomix"] = self.__read_field("int", fid, "bottomix", [nx+2, ny+2])
        fields["leftiy"]   = self.__read_field("int", fid, "leftiy",   [nx+2, ny+2])
        fields["rightiy"]  = self.__read_field("int", fid, "rightiy",  [nx+2, ny+2])
        fields["topiy"]    = self.__read_field("int", fid, "topiy",    [nx+2, ny+2])
        fields["bottomiy"] = self.__read_field("int", fid, "bottomiy", [nx+2, ny+2])
        
        fields["region"]      = self.__read_field("int", fid, "region",     [nx+2, ny+2, 3])
        fields["nnreg"]       = self.__read_field("int", fid, "nnreg",      [3])
        fields["resignore"]   = self.__read_field("int", fid, "resignore",  [nx+2, ny+2, 2])
        fields["periodic_bc"] = self.__read_field("int", fid, "periodic_bc", [1])
        
        fields["pbs"]  = self.__read_field("float", fid, "pbs",  [nx+2, ny+2, 2])

        def version_to_num(v):
            return int(v.replace(".", ""))
        
        if version_to_num(version) >= version_to_num("03.000.008"):
            fields["redef_gmtry"] = self.__read_field("int", fid, "redef_gmtry", [1])
            fields["wbbl"] = self.__read_field("float", fid, "wbbl", [nx+2, ny+2, 4])
            fields["cflags"] = self.__read_field("int", fid, "cflags", [nx+2, ny+2, 5])
            fields["cell_width"] = self.__read_field("float", fid, "cell_width", [nx+2, ny+2])
            fields["cell_height"] = self.__read_field("float", fid, "cell_height", [nx+2, ny+2])

        fields["parg"] = self.__read_field("float", fid, "parg", [100])

        # Close file
        fid.close()

        self.gmtry = fields

        
    def write_b2fstate(self, filename, label):
        """Write b2fstate file for use by B2.5."""

        if not self.state:
            print("No b2fstate is stored. Exiting write_b2fstate.")
        
        try:
            fid = open(filename, "w")
        except OSError as e:
            raise RuntimeError(f"Could not open file for writing: {e}")

        # --- Version ---
        version = self.state["version"]
        print(f"write_b2fstate -- file version {version}")
        VERSION = f"VERSION{version} Written from Python"
        fid.write(VERSION + "\n")

        # --- Dimensions ---
        nx = self.state["dim"][0] - 2
        ny = self.state["dim"][1] - 2
        ns = self.state["dim"][2]
        self.__write_ifield(fid, "nx,ny,ns", [nx, ny, ns])

        # --- Label ---
        self.__write_sfield(fid, "label", label)

        # --- Charges ---
        self.__write_rfield(fid, "zamin", self.state["zamin"])
        self.__write_rfield(fid, "zamax", self.state["zamax"])
        self.__write_rfield(fid, "zn   ", self.state["zn"])
        self.__write_rfield(fid, "am   ", self.state["am"])

        # --- State variables ---
        self.__write_rfield(fid, "na",     self.state["na"])
        self.__write_rfield(fid, "ne",     self.state["ne"])
        self.__write_rfield(fid, "ua",     self.state["ua"])
        self.__write_rfield(fid, "uadia",  self.state["uadia"])
        self.__write_rfield(fid, "te",     self.state["te"])
        self.__write_rfield(fid, "ti",     self.state["ti"])
        self.__write_rfield(fid, "po",     self.state["po"])

        # --- Fluxes ---
        self.__write_rfield(fid, "fna",    self.state["fna"])
        self.__write_rfield(fid, "fhe",    self.state["fhe"])
        self.__write_rfield(fid, "fhi",    self.state["fhi"])
        self.__write_rfield(fid, "fch",    self.state["fch"])
        self.__write_rfield(fid, "fch_32", self.state["fch_32"])
        self.__write_rfield(fid, "fch_52", self.state["fch_52"])
        self.__write_rfield(fid, "kinrgy", self.state["kinrgy"])
        self.__write_rfield(fid, "time",   self.state["time"])
        self.__write_rfield(fid, "fch_p",  self.state["fch_p"])

        # --- Extra fields if version >= 03.000.005 ---
        def version_to_num(v): return int(v.replace(".", ""))
        if version_to_num(version) >= version_to_num("03.000.005"):
            self.__write_rfield(fid, "fna_mdf",    self.state["fna_mdf"])
            self.__write_rfield(fid, "fhe_mdf",    self.state["fhe_mdf"])
            self.__write_rfield(fid, "fhi_mdf",    self.state["fhi_mdf"])
            self.__write_rfield(fid, "fna_fcor",   self.state["fna_fcor"])
            self.__write_rfield(fid, "fna_nodrift",self.state["fna_nodrift"])
            self.__write_rfield(fid, "fna_he",     self.state["fna_he"])
            self.__write_rfield(fid, "fnaPSch",    self.state["fnaPSch"])
            self.__write_rfield(fid, "fhePSch",    self.state["fhePSch"])
            self.__write_rfield(fid, "fhiPSch",    self.state["fhiPSch"])
            self.__write_rfield(fid, "fna_eir",    self.state["fna_eir"])
            self.__write_rfield(fid, "fne_eir",    self.state["fne_eir"])
            self.__write_rfield(fid, "fhe_eir",    self.state["fhe_eir"])
            self.__write_rfield(fid, "fhi_eir",    self.state["fhi_eir"])
            self.__write_rfield(fid, "fna_32",     self.state["fna_32"])
            self.__write_rfield(fid, "fna_52",     self.state["fna_52"])
            self.__write_rfield(fid, "fni_32",     self.state["fni_32"])
            self.__write_rfield(fid, "fni_52",     self.state["fni_52"])
            self.__write_rfield(fid, "fne_32",     self.state["fne_32"])
            self.__write_rfield(fid, "fne_52",     self.state["fne_52"])
            self.__write_rfield(fid, "fchdia",     self.state["fchdia"])
            self.__write_rfield(fid, "fchin",      self.state["fchin"])
            self.__write_rfield(fid, "fchvispar",  self.state["fchvispar"])
            self.__write_rfield(fid, "fchvisper",  self.state["fchvisper"])
            self.__write_rfield(fid, "fchvisq",    self.state["fchvisq"])
            self.__write_rfield(fid, "fchinert",   self.state["fchinert"])
            self.__write_rfield(fid, "vaecrb",     self.state["vaecrb"])
            self.__write_rfield(fid, "vadia",      self.state["vadia"])
            self.__write_rfield(fid, "wadia",      self.state["wadia"])
            self.__write_rfield(fid, "veecrb",     self.state["veecrb"])
            self.__write_rfield(fid, "vedia",      self.state["vedia"])
            self.__write_rfield(fid, "floe_noc",   self.state["floe_noc"])
            self.__write_rfield(fid, "floi_noc",   self.state["floi_noc"])

        fid.close()

    
    def write_b2fgmtry(self, filename, label):
        """
        Write b2fgmtry file for use by B2.5.
        
        Parameters
        ----------
        filename : str
           Path to the output file
        label : str
           Label string
        """

        if not self.gmtry:
            print("No b2fgmtry is stored. Exiting write_b2fgmtry.")
            return

        gmtry = self.gmtry
        
        try:
            fid = open(filename, "w")
        except OSError as e:
            raise RuntimeError(f"Could not open {filename}: {e}")

        # --- Write version header
        version = gmtry["version"]
        VERSION = f"VERSION{version} Written from Python"
        fid.write(f"{VERSION}\n")

        # --- Dimensions nx, ny
        nx = gmtry["vol"].shape[0] - 2
        ny = gmtry["vol"].shape[1] - 2
        self.__write_ifield(fid, "nx,ny", [nx, ny])
        
        # --- Label
        self.__write_sfield(fid, "label", label)
        
        # --- Symmetry information
        self.__write_ifield(fid, "isymm", gmtry["isymm"])
        
        # --- Geometry variables
        self.__write_rfield(fid, "crx", gmtry["crx"])
        self.__write_rfield(fid, "cry", gmtry["cry"])
        self.__write_rfield(fid, "fpsi", gmtry["fpsi"])
        self.__write_rfield(fid, "ffbz", gmtry["ffbz"])
        self.__write_rfield(fid, "bb", gmtry["bb"])
        self.__write_rfield(fid, "vol", gmtry["vol"])
        self.__write_rfield(fid, "hx", gmtry["hx"])
        self.__write_rfield(fid, "hy", gmtry["hy"])
        self.__write_rfield(fid, "qz", gmtry["qz"])
        self.__write_rfield(fid, "qc", gmtry["qc"])
        self.__write_rfield(fid, "gs", gmtry["gs"])
        
        # --- Other geometrical parameters
        self.__write_ifield(fid, "nlreg", gmtry["nlreg"])
        self.__write_ifield(fid, "nlxlo", gmtry["nlxlo"])
        self.__write_ifield(fid, "nlxhi", gmtry["nlxhi"])
        self.__write_ifield(fid, "nlylo", gmtry["nlylo"])
        self.__write_ifield(fid, "nlyhi", gmtry["nlyhi"])
        self.__write_ifield(fid, "nlloc", gmtry["nlloc"])
        
        self.__write_ifield(fid, "nncut", gmtry["nncut"])
        self.__write_ifield(fid, "leftcut", gmtry["leftcut"])
        self.__write_ifield(fid, "rightcut", gmtry["rightcut"])
        self.__write_ifield(fid, "topcut", gmtry["topcut"])
        self.__write_ifield(fid, "bottomcut", gmtry["bottomcut"])
        
        self.__write_ifield(fid, "leftix", gmtry["leftix"])
        self.__write_ifield(fid, "rightix", gmtry["rightix"])
        self.__write_ifield(fid, "topix", gmtry["topix"])
        self.__write_ifield(fid, "bottomix", gmtry["bottomix"])
        self.__write_ifield(fid, "leftiy", gmtry["leftiy"])
        self.__write_ifield(fid, "rightiy", gmtry["rightiy"])
        self.__write_ifield(fid, "topiy", gmtry["topiy"])
        self.__write_ifield(fid, "bottomiy", gmtry["bottomiy"])
        
        self.__write_ifield(fid, "region", gmtry["region"])
        self.__write_ifield(fid, "nnreg", gmtry["nnreg"])
        self.__write_ifield(fid, "resignore", gmtry["resignore"])
        self.__write_ifield(fid, "periodic_bc", gmtry["periodic_bc"])
        
        self.__write_rfield(fid, "pbs", gmtry["pbs"])

        def version_to_num(v):
            return int(v.replace(".", ""))
        if version_to_num(version) >= version_to_num("03.000.008"):
            self.__write_ifield(fid, "redef_gmtry", gmtry["redef_gmtry"])
            self.__write_rfield(fid, "wbbl", gmtry["wbbl"])
            self.__write_ifield(fid, "cflags", gmtry["cflags"])
            self.__write_rfield(fid, "cell_width", gmtry["cell_width"])
            self.__write_rfield(fid, "cell_height", gmtry["cell_height"])

        self.__write_rfield(fid, "parg", gmtry["parg"])
        
        # --- Close file
        fid.close()
        
    def __read_field(self, my_type, fid, fieldname, dims):
        """
        Auxiliary routine to read real fields from B2.5 b2f* files.
        
        Parameters
        ----------
        fid : file object
           Open file handle (text mode).
        fieldname : str
           Identifier to search for in the file.
        dims : tuple of int
           Dimensions of the field to read.
        
        Returns
        -------
        field : numpy.ndarray
           Array of real values read from the file.
        """
        found = False
        
        # Search the file until identifier 'fieldname' is found
        for line in fid:
            if fieldname in line:
                found = True
                break

        if(not found):
            raise EOFError(f"EOF reached without finding {fieldname}.")
            
        # Consistency check
        parts = line.split()
        try:
            numin = int(parts[2])   # MATLAB's '%*s %*s %d' skips 2 words, reads 3rd as int
        except (IndexError, ValueError):
            raise ValueError(f"Could not parse number of elements from line: {line.strip()}")
        
        if numin != np.prod(dims):
            raise ValueError("read_rfield: inconsistent number of input elements.")
            
        # Read the data
        data = []

        while len(data) < numin:
            chunk = fid.readline()
            if not chunk:
                raise EOFError("Unexpected EOF while reading data values.")
            if(my_type.lower() == "int"):
                data.extend([int(x) for x in chunk.split()])
            elif(my_type.lower() == "float"):
                data.extend([float(x) for x in chunk.split()])
            
        field = np.array(data[:numin])

        # Reshape if needed
        if my_type.lower() != "int":
            if len(dims) > 1:
                field = field.reshape(dims,order='F')
                    
        return field
    
    def __write_ifield(self, fid, name, values):
        """Write integer field to file."""
        values = np.atleast_1d(values).ravel()
        fid.write("*cf:    int ")
        fid.write(f'%16d'%len(values))
        fid.write(f'    {name:<32}\n')
        for i in range(0, len(values), 12):
            line = "".join(f"{v:11d}" for v in values[i:min(i+12,len(values))])
            fid.write(line + "\n")
   
        
    def __write_rfield(self, fid, name, values):
        """Write real (float) field to file."""
        values = np.atleast_1d(values).ravel(order='F')
        fid.write("*cf:    real")
        fid.write(f'%16d'%len(values))
        fid.write(f'    {name:<32}\n')
        # Write values in scientific notation, 6 per line
        for i in range(0, len(values), 6):
            line = ""
            for v in values[i:min(i+6,len(values))]:
                formatted_number = f"{v: 19.13E}"
                mantissa, exponent = formatted_number.split('E')
                formatted_exponent = f"{int(exponent):+04d}"
                final_num = f"{mantissa}E{formatted_exponent}"
                line = line+" "+final_num 
            fid.write(line + "\n")

    def __write_sfield(self, fid, name, value):
        """Write string field to file."""
        fid.write("*cf:    char")
        fid.write(f'%16d'%120)
        fid.write(f'    {name:<32}\n')
        fid.write(f" {value:<120}\n")
