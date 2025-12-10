import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

def plot_multi_roc(fpr_list, tpr_list, auc_list, tar20_list, tar30_list, labels):
    palette = qualitative.Set1
    n_colors = len(palette)
    line_styles   = ["solid", "dash", "dot", "dashdot", "longdash"]
    marker_symbols = ["circle", "x", "triangle-up", "diamond", "star", "hexagon"]
    fig = go.Figure()

    for i, (fpr, tpr, auc, tar20, tar30, lbl) in enumerate(
        zip(fpr_list, tpr_list, auc_list, tar20_list, tar30_list, labels)
    ):
        color     = palette[i % n_colors]
        if str(palette[i % n_colors]) == "rgb(255,255,51)":
            color = "rgb(255,0,255)"
        # color     = custom_colors[i % len(custom_colors)]
        dash      = line_styles[i % len(line_styles)]
        symbol    = marker_symbols[i % len(marker_symbols)]

        # ROC curve line
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{lbl} (AUC={auc:.2f}%)",
            line=dict(color=color, dash=dash, width=2)
        ))

    # diagonal chance line
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ))

    fig.update_layout(#margin=dict(l=10, r=2, t=30, b=10),
    xaxis=dict(
        title=dict(
            text="False Positive Rate (1- Specificity)",
            font=dict(size=20)
        )
    ),
    yaxis=dict(
        title=dict(
            text="True Positive Rate (Sensitivity)",
            font=dict(size=20)
        )
    ),
    width=1000, height=1000,
    legend=dict(
        x=0.995,
        y=0.005,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.5)",
        font=dict(family="sans-serif", size=18),
        tracegroupgap=5
    ),
    plot_bgcolor="white",
    paper_bgcolor="white"
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1, 
        linecolor="black",
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
    )
    fig.update_yaxes(scaleanchor="y", scaleratio=1)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    fig.write_image("Path to save ROC")

if __name__ == "__main__":
    ds          = np.load("Path to infered numpy matrix")
    final_llama = np.load("Path to infered numpy matrix")
    final_ds    = np.load("Path to infered numpy matrix")
    final_open  = np.load("Path to infered numpy matrix")
    gbc         = np.load("Path to infered numpy matrix")
    lgbm        = np.load("Path to infered numpy matrix")
    llama       = np.load("Path to infered numpy matrix")
    lr          = np.load("Path to infered numpy matrix")
    openbio     = np.load("Path to infered numpy matrix")
    RFC         = np.load("Path to infered numpy matrix")
    xgb         = np.load("Path to infered numpy matrix")

    fpr_list   = [ds["fpr"], final_llama["fpr"], final_ds["fpr"], final_open["fpr"], gbc["fpr"], lgbm["fpr"], llama["fpr"], lr["fpr"], openbio["fpr"], RFC["fpr"], xgb["fpr"]]
    tpr_list   = [ds["tpr"], final_llama["tpr"], final_ds["tpr"], final_open["tpr"], gbc["tpr"], lgbm["tpr"], llama["tpr"], lr["tpr"], openbio["tpr"], RFC["tpr"], xgb["tpr"]]
    auc_list   = [ds["auc"]*100, final_llama["auc"]*100, final_ds["auc"]*100, final_open["auc"]*100, gbc["auc"]*100, lgbm["auc"]*100, llama["auc"]*100, lr["auc"]*100, openbio["auc"]*100, RFC["auc"]*100, xgb["auc"]*100]
    tar20_list = [ds["tar20"], final_llama["tar20"], final_ds["tar20"], final_open["tar20"], gbc["tar20"], lgbm["tar20"], llama["tar20"], lr["tar20"], openbio["tar20"], RFC["tar20"], xgb["tar20"]]
    tar30_list = [ds["tar30"], final_llama["tar30"], final_ds["tar30"], final_open["tar30"], gbc["tar30"], lgbm["tar30"], llama["tar30"], lr["tar30"], openbio["tar30"], RFC["tar30"], xgb["tar30"]]
    labels     = ['DeepSeek R1 distill Qwen 32B','MIRACLE (Llama 3.3 70B-Instruct)','MIRACLE (DeepSeek R1 distill Qwen 32B)','MIRACLE (OpenBioLLM-70B)','Gradient Boosting Classifier','Light GBM','Llama 3.3 70B-Instruct', 'Logistic Regression','OpenBioLLM-70B','Random Forest Classifier','XGBoost']

    plot_multi_roc(fpr_list, tpr_list, auc_list, tar20_list, tar30_list, labels)
