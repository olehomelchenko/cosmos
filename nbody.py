import streamlit as st
from nbody_util import Universe
import pandas as pd
from io import StringIO
import numpy as np
import plotly.express as px
from datetime import datetime

import matplotlib
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import streamlit.components.v1 as comp

matplotlib.use("TkAgg")

NUM = st.sidebar.number_input

G = 1  # NUM("G", value=1, min_value=1, step=1)
N = NUM("Number of bodies", min_value=2, value=3)
dt = NUM("DT", min_value=0.00001, value=0.01, step=0.5)
pos_std = 1  # NUM("Pos STD", value=1, min_value=1, step=10)
mas_std = 1  # NUM("Mas STD", value=1, min_value=1, step=10)
acc_std = 1  # NUM("Acc STD", value=1, min_value=1, step=10)
scale = NUM("Scale", value=100, min_value=1, step=1)
number_of_frames = NUM("Num of frames", value=1000, min_value=1)
frame_crop = NUM("Frame Crop", value=100, min_value=1, step=10)

ex_string = """0,0,1000,0,0
0,10,40,10,0
0,50,100,-5,0

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
st_com = st.empty()

st_obj_head = st.empty()

history_df = u.get_obj_df()

time_start = datetime.now()
while u.current_iteration < number_of_frames:
    u.obj = u.calc_positions(reset_com=True)
    if u.current_iteration % frame_crop == 0:
        # st_altair.altair_chart(u.plot_altair((-scale, scale), (-scale, scale)))

        history_df = pd.concat([history_df, u.get_obj_df()])
        st_iter.text(u.current_iteration)

time_end = datetime.now()

st.text(f"Generating {u.current_iteration} iterations for {(time_end - time_start)}")

st.dataframe(history_df)

df = history_df[history_df["iter"] == 0]

fig = plt.figure()

scat = plt.scatter(x=df["x"], y=df["y"], s=df["m"])


def update_plot(i):
    df = history_df[history_df["iter"] == i]
    x = df["x"]
    y = df["y"]
    st.text(df)
    scat.set_offsets(np.c_[x, y])
    return scat


ani = FuncAnimation(fig, update_plot, frames=history_df["iter"].unique(), blit=False)

# FFwriter = FFMpegWriter()
# ani.save('animation.mp4', writer = FFwriter)


comp.html(ani.to_jshtml(), height=800)

# fig = px.scatter(
#     history_df,
#     x="x",
#     y="y",
#     animation_frame="iter",
#     size="m",
#     range_x=(-scale, scale),
#     range_y=(-scale, scale),
#     width=700,
#     height=700,
# )

# st.plotly_chart(fig)
