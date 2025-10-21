import numpy as np
from pathlib import Path

# Class containing a triangular mesh defined by nodes cells and links,
#    primarily for Eirene mesh
class triangle_mesh:
    def __init__(self, filepath=None):
        self.incenter = None
        if (filepath):
            self.read_ft33(filepath / Path("fort.33"))
            self.read_ft34(filepath / Path("fort.34"))
            self.read_ft35(filepath / Path("fort.35"))

    def read_ft33(self, filename):
        """
        Read fort.33-files (triangle nodes). Converts to SI units (m).
        
        Parameters
        ----------
        filename : str
           Path to the fort.33 file.
        
        """
        try:
            with open(filename, "r") as fid:
                print("read_ft33: assuming ntrfrm = 0.")
                ntrfrm = 0
                
                # number of nodes
                nnodes = int(fid.readline().strip())
                nodes = np.zeros((nnodes, 2), dtype=float)

                if ntrfrm == 0:
                    # read x coordinates
                    x = [list(map(float,next(fid).strip().split())) for _ in range(int(np.ceil(nnodes/4)))]
                    # read y coordinates
                    y = [list(map(float,next(fid).strip().split())) for _ in range(int(np.ceil(nnodes/4)))]
                    nodes[:, 0] = self.__flatten(x)
                    nodes[:, 1] = self.__flatten(y)
                else:
                    raise ValueError("read_ft33: wrong ntrfrm.")
                
                # convert cm → m
                nodes *= 1e-2
            self.nodes=nodes    
            
        except OSError as e:
            raise RuntimeError(f"Could not open file {filename}: {e}")

        
    def read_ft34(self, filename):
        """
        Read fort.34-files (nodes composing each triangle).
        
        Parameters
        ----------
        filename : str
           Path to the fort.34 file.
        """
        try:
            with open(filename, "r") as f:
                # number of triangles
                ntria = int(f.readline().strip())
                cells = np.zeros((ntria, 3), dtype=int)
                
                for i in range(ntria):
                    # read 4 integers, skip the first, keep the last 3
                    data = list(map(int, f.readline().split()))
                    if len(data) != 4:
                        raise ValueError(f"Line {i+2} does not contain 4 integers.")
                    cells[i, :] = data[1:]
                    
            self.cells=cells
                
        except OSError as e:
            raise RuntimeError(f"Could not open file {filename}: {e}")


    def read_ft35(self, filename):
        """
        Read fort.35-files (triangle data).
        
        Parameters
        ----------
        filename : str
           Path to the fort.35 file.
        
        Returns
        -------
        links : dict of ndarrays
        Dictionary containing:
           - 'nghbr': (ntria, 3) neighbor triangle indices
           - 'side' : (ntria, 3) side numbers
           - 'cont' : (ntria, 3) connectivity info
           - 'ixiy' : (ntria, 2) ix/iy indices
        """
        try:
            with open(filename, "r") as f:
                # number of triangles
                ntria = int(f.readline().strip())
                
                links = {
                    "nghbr": np.zeros((ntria, 3), dtype=int),
                    "side":  np.zeros((ntria, 3), dtype=int),
                    "cont":  np.zeros((ntria, 3), dtype=int),
                    "ixiy":  np.zeros((ntria, 2), dtype=int),
                }
                
                for i in range(ntria):
                    # read 12 integers
                    data = list(map(int, f.readline().split()))
                    if len(data) != 12:
                        raise ValueError(f"Line {i+2} does not contain 12 integers.")
                    
                    # MATLAB indexing (2:3:8, etc.) → Python slices
                    links["nghbr"][i, :] = data[1:8:3]   # indices 2,5,8 in MATLAB
                    links["side"][i, :]  = data[2:9:3]  # indices 3,6,9
                    links["cont"][i, :]  = data[3:10:3]  # indices 4,7,10
                    links["ixiy"][i, :]  = data[10:12]   # indices 11,12
                    
            self.links=links
                
        except OSError as e:
            raise RuntimeError(f"Could not open file {filename}: {e}")

    def calc_incenter(self, recalculate=False):
        if(recalculate or not self.incenter):
            # subtract off one because cells is 1 based
            x = self.nodes[self.cells-1, 0] 
            y = self.nodes[self.cells-1, 1]
            a = np.sqrt((x[:,2]-x[:,1])**2 + (y[:,2]-y[:,1])**2)
            b = np.sqrt((x[:,2]-x[:,0])**2 + (y[:,2]-y[:,0])**2)
            c = np.sqrt((x[:,1]-x[:,0])**2 + (y[:,1]-y[:,0])**2)
            mag = a+b+c
            Ix = (a*x[:,0]+b*x[:,1]+c*x[:,2])/mag
            Iy = (a*y[:,0]+b*y[:,1]+c*y[:,2])/mag
            self.incenter = np.array([Ix,Iy]).T
    
    def write_ft33(self, file):
        """
        Write fort.33-files (triangle nodes). Converts to EIRENE units (cm).
        
        Parameters
        ----------
        file : str
           Output filename
        """

        try:
            fid = open(file, "w")
        except OSError as e:
            raise RuntimeError(f"Could not open file {file}: {e}")
        
        print("write_ft33: assuming ntrfrm = 0.")
        ntrfrm = 0

        nodes = self.nodes
        # Convert to cm
        nodes = np.array(nodes) * 1e2
        
        # Write number of nodes
        fid.write(f"{nodes.shape[0]:12d}\n")

        if ntrfrm == 0:
            # Write x and y columns separately
            for col in range(2):
                coldata = nodes[:, col]
                for i, val in enumerate(coldata, start=1):
                    fid.write(f"{val:19.8E}")
                    if i % 4 == 0:  # 4 numbers per line
                        fid.write("\n")
                    if len(coldata) % 4 != 0:  # end line if not multiple of 4
                        fid.write("\n")
        else:
            raise ValueError("write_ft33: wrong ntrfrm.")
                    
        fid.close()

        
    def write_ft34(self, file):
        """
        Write fort.34-files (nodes composing each triangle).

        Parameters
        ----------
        file : str
           Output filename
        """

        try:
            fid = open(file, "w")
        except OSError as e:
            raise RuntimeError(f"Could not open file {file}: {e}")

        cells = self.cells
        # number of triangles
        ntria = cells.shape[0]
        fid.write(f"{ntria:12d}\n")

        # write each triangle: index + 3 node indices
        for i in range(ntria):
            # i+1 because MATLAB indices start at 1
            fid.write(f"{i+1:6d} {cells[i,0]:7d} {cells[i,1]:5d} {cells[i,2]:5d}\n")

        fid.close()


    def write_ft35(self, file):
        """
        Write fort.35-files (triangle data).
        
        Parameters
        ----------
        file : str
           Output filename
        links : object or dict-like
           Must contain the fields:
           - nghbr (ntria, 3)
           - side  (ntria, 3)
           - cont  (ntria, 3)
           - ixiy  (ntria, 2)
        """

        links = self.links
        try:
            fid = open(file, "w")
        except OSError as e:
            raise RuntimeError(f"Could not open file {file}: {e}")

        ntria = links["nghbr"].shape[0]
        fid.write(f"{ntria:12d}\n")

        for i in range(ntria):
            line = (
                f"{i+1:6d}"
                f" {links['nghbr'][i,0]:7d} {links['side'][i,0]:5d} {links['cont'][i,0]:5d}"
                f" {links['nghbr'][i,1]:9d} {links['side'][i,1]:5d} {links['cont'][i,1]:5d}"
                f" {links['nghbr'][i,2]:9d} {links['side'][i,2]:5d} {links['cont'][i,2]:5d}"
                f" {links['ixiy'][i,0]:9d} {links['ixiy'][i,1]:5d}\n"
            )
            fid.write(line)

        fid.close()

            
    def __flatten(self, xss):
        return [x for xs in xss for x in xs]
