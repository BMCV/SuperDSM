import os
import subprocess

cpu_vendor = 'unknown'
try:
    vendor_info = subprocess.check_output('cat /proc/cpuinfo |grep -i vendor', shell=True).decode().strip()

    if 'AuthenticAMD' in vendor_info:
        cpu_vendor = 'amd'

    elif 'GenuineIntel' in vendor_info:
        cpu_vendor = 'intel'

except:
    pass


if cpu_vendor == 'unknown':
    print("""
    ***************************************************************************************
    * Failed to determine CPU vendor.                                                     *
    *                                                                                     *
    * If this is an AMD CPU, please set MKL_DEBUG_CPU_TYPE=5 and use MKL 2020.0 for BLAS. *
    * For details, see: https://github.com/BMCV/SuperDSM#performance-considerations       *
    ***************************************************************************************
    """)


if cpu_vendor == 'amd':

    os.environ['MKL_DEBUG_CPU_TYPE'] = '5'


if cpu_vendor in ('unknown', 'intel'):

    # Force MKL on the same execution path which it uses for AMD CPUs.
    # This is slower, but establishes reproducibility.
    os.environ['MKL_DEBUG_CPU_TYPE'] = '1'
