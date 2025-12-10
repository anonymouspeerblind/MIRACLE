import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    pastel_scale = [
        [0.0, "#ffffff"],   # low values = white
        [1.0, "#f1bc8d"]    # high values = pastel blue
    ]
    human_annotation = pd.read_csv("Path to surgeons' annotation csv")['SurgeonLabel'].tolist()
    ground_truth = pd.read_csv("Path to test split csv")['Target'].tolist()
    
    cm = confusion_matrix(ground_truth, human_annotation, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    # Create matrix with counts + percentages
    cm_text = [
        [f"TN: {tn}<br>{tn/total:.1%}", f"FP: {fp}<br>{fp/total:.1%}"],
        [f"FN: {fn}<br>{fn/total:.1%}", f"TP: {tp}<br>{tp/total:.1%}"]
    ]

    # Plot confusion matrix with annotations
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["No Postoperative Complication", "Postoperative Complication"],
        y=["No Postoperative Complication", "Postoperative Complication"],
        annotation_text=cm_text,
        colorscale=pastel_scale,
        showscale=True
    )
    for anno in fig.layout.annotations:
        anno.font.size = 14
        anno.font.family = "sans-serif"

    fig.update_layout(
        xaxis_title="Surgeon Annotation",
        yaxis_title="Actual Label",
        xaxis=dict(
            side="bottom",
        tickfont=dict(size=10),
        title=dict(font=dict(size=18), standoff=10)
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            tickangle=-90,
            title=dict(font=dict(size=18), standoff=10)
        ),
        title_font=dict(size=20) 
    )
    fig.data[0].colorbar.tickfont = dict(
        family="sans-serif",
        size=12
    )
    fig.write_image("Confusion_matrix.png")
