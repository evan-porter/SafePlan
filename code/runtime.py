import sys
from glob import glob
from connect import get_current

# All that is placed in the script. Remainder can be imported from file if needed.
DATA_PATH = 'Path\To\RayStation\HeadNode\'
IMPORT_PATH = 'Path\To\ParentOfDatabases\'

sys.path.append(IMPORT_PATH)
from interface import UserInterface
from backend import BackendManager

# Use the newest databases
ref_dbs = glob(DATA_PATH + 'dbs\\ReferenceConstraints*.db')
ref_dbs.sort()
ref_db_path = ref_dbs[-1]
print(f'Using Reference DB: {ascii(ref_db_path)}')

log_dbs = glob(DATA_PATH + 'dbs\\ChangeLog*.db')
log_dbs.sort()
log_db_path = log_dbs[-1]
print(f'Writing to Log DB: {ascii(ref_db_path)}')

# Create a backend instance and run the UI
bm = BackendManager(get_current, ref_db_path, log_db_path)
ui = UserInterface(bm)

# User should use one of these per script in RayStation
ui.on_import()
ui.on_export()
ui.on_ICC()
ui.on_MD()
