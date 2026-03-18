import pyvelora as pv

# ---- vector creation ----
v1 = pv.Vector([1, 2, 3])
print(v1)  # Vector([1. 2. 3.])
v2 = pv.Vector([4, 5, 6], type="polar", degrees=True)
print(v2)  # Vector([-4.8985872e-16  4