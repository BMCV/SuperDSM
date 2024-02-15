import os
import subprocess

failed = False
try:
    vendor_info = subprocess.check_output('cat /proc/cpuinfo |grep -i vendor', shell=True).decode().strip()

    if 'AuthenticAMD' in vendor_info:
        os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

    elif 'GenuineIntel' in vendor_info:
        del os.environ['MKL_DEBUG_CPU_TYPE']

    else:
        failed = True

except:
    failed = True

if failed:
    print("""
    ***************************************************************************************
    * Failed to determine CPU vendor.                                                     *
    *                                                                                     *
    * If this is an AMD CPU, please set MKL_DEBUG_CPU_TYPE=5 and use MKL 2020.0 for BLAS. *
    * For details, see: https://github.com/BMCV/SuperDSM#performance-considerations       *
    ***************************************************************************************
    """)
