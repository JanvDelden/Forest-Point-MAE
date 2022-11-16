import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

def explore(points, string="", show=True, write=False, i=None, size_max=6, cont_color=False, colors=None):  
    labels_exist = points.shape[1] == 4
    rgb_exists = points.shape[1] == 6
    if labels_exist:
        labels = points[:,3]
        if not cont_color:
            labels = labels.astype("str")
        d = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2], "label": labels}
        df = pd.DataFrame(data=d)
    elif rgb_exists:
        d = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2], "R": points[:,3], "G": points[:,4],"B": points[:,5]}
        df = pd.DataFrame(data=d)
    else:
        df = pd.DataFrame(data=points, columns=["x", "y", "z"])
    size = np.ones(len(points))
    if not rgb_exists:
        fig = px.scatter_3d(df, x='x', y='y', z='z', color="label" if labels_exist else None, \
                            size=size, opacity=0, size_max=size_max, template="plotly_dark", color_continuous_scale="agsunset")
    else:
        config = go.Scatter3d(x=df.x, y=df.y, z=df.z,mode='markers',
                      marker=dict(size=1,
                                  color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(df.R.values, df.G.values, df.B.values)],
                                  opacity=0.9,))
        data = [config]
        fig = go.Figure(data=data)

    #fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_scenes(aspectmode='data')
    fig.update_traces(marker=dict(line=dict(width=4, color='Black')), selector=dict(mode='markers'))
    if show:
        fig.show()
    if write:
        fig.write_html(string + str(i) + ".html")

""" import numpy as np
path = "/user/jschnei2/data/trees/tmp/-42-13-31-2.npy"
points = np.load(path)

explore(points) """

def patches_format(arr, scaler=[10, 10, 22.5]):
    assert len(arr.shape) == 3
    patch_idx = np.repeat(np.arange(arr.shape[0]), repeats=arr.shape[1])
    arr = arr.reshape(-1, 3)
    if scaler:
        arr[:, 2] += 1
        arr = arr * np.array(scaler)
    return np.hstack([arr, patch_idx[:, np.newaxis]])


