import json
import plotly.graph_objects as go

if __name__ == "__main__":
    with open("<Path to LLM as a judge file for LLM 1>", "r") as js:
        scores1 = json.load(js)
    scores1_dict = {"Completely unaligned": 0, "Somewhat aligned": 0, "Completely aligned": 0}
    for record in scores1:
        if scores1[record] == 0:
            scores1_dict["Completely unaligned"] += 1
        elif scores1[record] == 1:
            scores1_dict["Somewhat aligned"] += 1
        else:
            scores1_dict["Completely aligned"] += 1
    
    with open("<Path to LLM as a judge file for LLM 2>", "r") as js:
        scores2 = json.load(js)
    scores2_dict = {"Completely unaligned": 0, "Somewhat aligned": 0, "Completely aligned": 0}
    for record in scores2:
        if scores2[record] == 0:
            scores2_dict["Completely unaligned"] += 1
        elif scores2[record] == 1:
            scores2_dict["Somewhat aligned"] += 1
        else:
            scores2_dict["Completely aligned"] += 1

    with open("<Path to LLM as a judge file for LLM 3>", "r") as js:
        scores3 = json.load(js)
    scores3_dict = {"Completely unaligned": 0, "Somewhat aligned": 0, "Completely aligned": 0}
    for record in scores3:
        if scores3[record] == 0:
            scores3_dict["Completely unaligned"] += 1
        elif scores3[record] == 1:
            scores3_dict["Somewhat aligned"] += 1
        else:
            scores3_dict["Completely aligned"] += 1
    
    categories   = list(scores1_dict.keys())
    frequencies1 = list(scores1_dict.values())
    frequencies2 = list(scores2_dict.values())
    frequencies3 = list(scores3_dict.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=frequencies1,
        name="OpenBioLLM-70B",  # Name for the legend
        text=frequencies1,
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=frequencies2,
        name="Llama 3.3 70B-Instruct",  # Name for the legend
        text=frequencies2,
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=frequencies3,
        name="Deep Seek R1-distill Qwen-32B",  # Name for the legend
        text=frequencies3,
        textposition='outside'
    ))

    fig.update_layout(
        title="Human vs Multiple LLMs",
        xaxis_title="Alignment",
        yaxis_title="Frequency",
        barmode='group',  # Group the bars next to each other
        height=600,  # Increase the height of the plot
        width=800,   # Increase the width of the plot
        margin=dict(t=50, b=100, l=50, r=50)
    )

    fig.write_image("Combined_human_vs_llm.png")