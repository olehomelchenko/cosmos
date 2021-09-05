import streamlit as st
from nbody_util import Universe
import numpy as np
import pandas as pd
from io import StringIO
import time
import matplotlib.pyplot as plt

NUM = st.sidebar.number_input

G = NUM("G", value=1, min_value=1, step=1)
N = NUM("Number of bodies", min_value=2, value=3)
dt = NUM("DT", min_value=0.00001, value=0.01, step=0.5)
pos_std = NUM("Pos STD", value=1, min_value=1, step=10)
mas_std = NUM("Mas STD", value=1, min_value=1, step=10)
acc_std = NUM("Acc STD", value=1, min_value=1, step=10)
scale = NUM("Scale", value=100, min_value=1, step=1)

ex_string = """0,0,1000,0,0
0,10,40,10,0
"""

ex_string = st.text_area("Manual Array of objects", value=ex_string)
ioo = StringIO(ex_string)
numpy_obj = pd.read_csv(ioo, sep=",", header=None).to_numpy()

u = Universe(
    G, N, dt=dt, pos_std=pos_std, mas_std=mas_std, acc_std=acc_std, obj=numpy_obj
)

st_iter = st.empty()

st_altair = st.empty()
st_borders = st.empty()

st_obj_head = st.empty()

while True:
    u.obj = u.calc_positions()
    st_altair.altair_chart(
        u.plot_altair(x_domain=[-scale, scale], y_domain=[-scale, scale])
    )
    st_iter.text(u.current_iteration)
    st_borders.text(u.borders)
    st_obj_head.table(u.obj[0:10])
    time.sleep(0.05)
