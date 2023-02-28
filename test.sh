#mpi: pmpjpe 77.8, pck 90.4, auc 55.2
python test.py --data mpi
#h36mp2 (17 joints): pmpjpe 50.9, mpjpe 93.4
python test.py --data h36mp2
#h36mp2 (14 joints): pmpjpe 55.1, mpjpe 101.6
python test.py --data h36mp2 --j14
#h36mp2 (17 joints): pmpjpe 54.7, mpjpe 96.3
python test.py --data h36mp1
#h36mp2 (17 joints): pmpjpe 59.2, mpjpe 105
python test.py --data h36mp1 --j14
#3dpw: pmpjpe 56.8, pve 143.2
python test.py --data 3dpw