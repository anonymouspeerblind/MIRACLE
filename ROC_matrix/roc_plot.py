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
        dash      = line_styles[i % len(line_styles)]
        symbol    = marker_symbols[i % len(marker_symbols)]

        # ROC curve line
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{lbl} (AUC={auc:.3f})",
            line=dict(color=color, dash=dash, width=2)
        ))

    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ))

    fig.update_layout(
        title="",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=1000, height=900,
        legend=dict(tracegroupgap=16)
    )
    fig.write_image("combined_ROC.png")

if __name__ == "__main__":
    ds          = np.load("Deepseek_bench_roc_values.npz")
    final_llama = np.load("final_model_roc_values_llama.npz")
    final_ds    = np.load("final_model_roc_values_remarks_dsr132b.npz")
    final_open  = np.load("final_model_roc_values_remarks_openbiollm.npz")
    gbc         = np.load("GBC_roc_values.npz")
    lgbm        = np.load("lgbm_roc_values.npz")
    llama       = np.load("llama_bench_roc_values.npz")
    lr          = np.load("LR_roc_values.npz")
    openbio     = np.load("openbio_bench_roc_values.npz")
    RFC         = np.load("RFC_roc_values.npz")
    xgb         = np.load("xgboost_roc_values.npz")

    fpr_list   = [ds["fpr"], final_llama["fpr"], final_ds["fpr"], final_open["fpr"], gbc["fpr"], lgbm["fpr"], llama["fpr"], lr["fpr"], openbio["fpr"], RFC["fpr"], xgb["fpr"]]
    tpr_list   = [ds["tpr"], final_llama["tpr"], final_ds["tpr"], final_open["tpr"], gbc["tpr"], lgbm["tpr"], llama["tpr"], lr["tpr"], openbio["tpr"], RFC["tpr"], xgb["tpr"]]
    auc_list   = [ds["auc"], final_llama["auc"], final_ds["auc"], final_open["auc"], gbc["auc"], lgbm["auc"], llama["auc"], lr["auc"], openbio["auc"], RFC["auc"], xgb["auc"]]
    tar20_list = [ds["tar20"], final_llama["tar20"], final_ds["tar20"], final_open["tar20"], gbc["tar20"], lgbm["tar20"], llama["tar20"], lr["tar20"], openbio["tar20"], RFC["tar20"], xgb["tar20"]]
    tar30_list = [ds["tar30"], final_llama["tar30"], final_ds["tar30"], final_open["tar30"], gbc["tar30"], lgbm["tar30"], llama["tar30"], lr["tar30"], openbio["tar30"], RFC["tar30"], xgb["tar30"]]
    labels     = ['DeepSeek R1 distill Qwen 32B','MIRACLE (Llama 3.3 70B-Instruct)','MIRACLE (DeepSeek R1 distill Qwen 32B)','MIRACLE (OpenBioLLM-70B)','Gradient Boosting Classifier','Light GBM','Llama 3.3 70B-Instruct', 'Logistic Regression','OpenBioLLM-70B','Random Forest Classifier','XGBoost']

    plot_multi_roc(fpr_list, tpr_list, auc_list, tar20_list, tar30_list, labels)