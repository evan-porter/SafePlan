# Authors
Evan Porter<sup>1</sup>, Nicolas Prionas<sup>1</sup>, Emily Hirata<sup>1</sup>, Alon Witztum<sup>1</sup>

*1. University of California San Francisco*

# Disclaimer
We offer no support for clinical use of this software and do not guarantee that this software will detect planning deviations in your clinic. Any users are solely responsible to commission and QA this software prior to clinical use.

# Steps to getting started
1. Dose constraints (`./constraints/*`) and ROI naming templates (`./templates/*`) should be updated to reflect clinical practice
2. The reference database can then be generated using `./code/rebuild_database.py`.
3. Specify the locations of clinical databases (`./code/backened.py` line 1603-1641)
4. Specify QA phantom name (`backend.py` line 1653)
5. Copy databases and code into the scripting directory on the RayStation head node
6. Create a new scirpting environment, run `pip install -r requirements.py`
7. Create scripts for each step by copying `./code/runtime.py` into the scripting creation
8. Update the location to reflect the code and database paths in `runtime.py`
9. Delete the on_(import / export / ICC / MD) lines except for the functionality of that script
10. Test the scripts, validate for clinical workflow and modify as needed.
11. Feel free to fork or submit PR to this project.

