
from mp_api.client import MPRester
import sys

# Define your Materials Project API key
API_KEY = None
if len(sys.argv) > 1:
    API_KEY = sys.argv[1]

# Initialize the MPRester client
with MPRester(API_KEY) as mpr:
    # Fetch the structure for the material with ID mp-19168
    structure = mpr.get_structure_by_material_id("mp-19168")

    # Write the structure to a POSCAR file
    structure.to(fmt="poscar", filename="POSCAR")

