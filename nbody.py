import streamlit as st
from nbody_util import Universe
import numpy as np
import pandas as pd
from io import StringIO
import time
import matplotlib.pyplot as plt

G = st.sidebar.number_input("G", value=1,min_value=1, step=1)
N = st.sidebar.number_input("Number of bodies", min_value=2, value=3)
dt = st.sidebar.number_input("DT", min_value=0.00001, value=0.01, step=0.5)
pos_std = st.sidebar.number_input("Pos STD", value=1, min_value=1, step=10)
mas_std = st.sidebar.number_input("Mas STD", value=1, min_value=1, step=10)
acc_std = st.sidebar.number_input("Acc STD", value=1, min_value=1, step=10)
scale = st.sidebar.number_input("Scale", value=100, min_value=1, step=1)

ex_string = """0,0,1000,0,0
0,10,40,10,0
"""

ex_string = st.text_area("Manual Array of objects", value=ex_string)
ioo = StringIO(ex_string)
numpy_obj = pd.read_csv(ioo, sep=",", header=None).to_numpy()

u = Universe(G, N, dt=dt, pos_std=pos_std, mas_std=mas_std, acc_std=acc_std, obj=numpy_obj)

# st_plot = st.pyplot(u.img[0])
st_iter = st.empty()

# fig, ax = plt.subplots()

st_altair = st.empty()
st_borders = st.empty()

st_obj_head = st.empty()

while True:
    u.obj = u.calc_positions()
    # u.plot_bodies(ax, xlim=(-10, 10), ylim=(-10, 10))
    st_altair.altair_chart(u.plot_altair(x_domain=[-scale, scale], y_domain=[-scale, scale]))
    # , xlim=(-10, 10), ylim=(-10, 10)
    # st_plot.pyplot(u.img[0])
    st_iter.text(u.current_iteration)
    st_borders.text(u.borders)
    st_obj_head.table(u.obj[0:10])
    time.sleep(0.05)