import streamlit as st
import pandas as pd
import numpy as np
import scipy.io
import networkx as nx
import plotly.graph_objects as go

from ui_components import apply_tactical_theme, COLOR_VOID, COLOR_WIRE, COLOR_STEEL, COLOR_ALERT
from roles_logic import run_all_role_methods

st.set_page_config(layout="wide")
apply_tactical_theme()

st.title("ROLES // CLASSIFICATION")
st.caption("SOCIAL ROLE ANALYSIS ENGINE")
st.info(
  "How to read: Colors = role type. Bigger nodes = more embedded in the network. "
  "Confidence = how much the different methods agree."
)


@st.cache_data
def load_adjacency():
  A = scipy.io.mmread("data/clandestine_network_example.mtx").tocsr()
  return A

A = load_adjacency()
G = nx.from_scipy_sparse_array(A)

@st.cache_data
def compute_roles(_A):
  return run_all_role_methods(_A)

results = compute_roles(A)

df_flow = results["df_flow"]
df_dist = results["df_dist"]
df_cent = results["df_cent"]
df_overlap = results["df_overlap"]

# flow based method for display; other methods are confidence indicators
def method_vote_for_node(i: int) -> dict:
  votes = {}

  votes["Flow"] = df_flow.loc[i, "role_name"]

  dr = df_dist.loc[i, "distance_role"]
  if dr == "High-degree Core":
    votes["Distance"] = "Core-like (high embeddedness)"
  elif dr == "Near high-degree core":
    votes["Distance"] = "Intermediate (moderate embeddedness)"
  elif dr == "Isolated":
    votes["Distance"] = "Extreme peripheral / near isolated"
  else:
    votes["Distance"] = "Peripheral (low embeddedness)"

  cr = df_cent.loc[i, "centrality_role_name"]
  if cr in ["Hub-like", "Influential"]:
    votes["Centrality"] = "Core-like (high embeddedness)"
  elif cr == "Bridge-like":
    votes["Centrality"] = "Intermediate (moderate embeddedness)"
  else:
    votes["Centrality"] = "Peripheral (low embeddedness)"

  deg = df_overlap.loc[i, "degree"]
  if deg <= 1:
    votes["Overlap"] = "Extreme peripheral / near isolated"
  elif deg <= np.median(df_overlap["degree"]):
    votes["Overlap"] = "Peripheral (low embeddedness)"
  else:
    votes["Overlap"] = "Intermediate (moderate embeddedness)"

  return votes

def confidence_for_node(i: int) -> float:
  votes = method_vote_for_node(i)
  final_role = votes["Flow"]
  agree = sum(v == final_role for v in votes.values())
  return agree / len(votes)

df_display = df_flow[["node","role_name","embeddedness_score","in_total","out_total","net_flow"]].copy()
df_display["confidence"] = [confidence_for_node(i) for i in df_display["node"]]

c1, c2, c3 = st.columns(3)
c1.metric(
  "Core-like members", int((df_display["role_name"] == "Core-like (high embeddedness)").sum()))
c2.metric("Peripheral members",
          int((df_display["role_name"]=="Peripheral (low embeddedness)").sum()))
c3.metric("Avg. confidence",
          f"{df_display['confidence'].mean():.0%}")

ROLE_COLORS = {
  "Core-like (high embeddedness)": "#e74c3c",
  "Intermediate (moderate embeddedness)": "#f39c12",
  "Peripheral (low embeddedness)": "#3498db",
  "Extreme peripheral / near isolated": "#7f8c8d"
}

def role_explanation(role):
  if "Core-like" in role:
    return "Highly connected and influential across multiple parts of the network."
  if "Intermediate" in role:
    return "Moderately connected; often acts as a link between core and peripheral members."
  if "Peripheral" in role:
    return "Limited interactions; participates in few network pathways."
  return "Very few connections; likely isolated or inactive."


def why_we_think_so(role: str) -> list[str]:
    if "Core-like" in role:
        return [
            "Appears in many multi-step connection routes (high network presence).",
            "Connects to members that are themselves highly active."
        ]
    if "Intermediate" in role:
        return [
            "Often sits between more central and more peripheral members.",
            "Acts as a connector between parts of the network."
        ]
    if "Peripheral" in role:
        return [
            "Has few direct ties compared with most members.",
            "Rarely part of longer connection routes."
        ]
    return [
        "Has very few (or no) observable ties in the data.",
        "Minimal network presence detected."
    ]


def plot_network(G, df_display):
  pos = nx.spring_layout(G, seed=42)

  edge_x, edge_y = [], []
  for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

  fig = go.Figure()
  fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line=dict(width=0.6, color="rgba(180,180,180,0.35)")))
  
  node_x, node_y, node_color, hover = [], [], [], []
  sizes = []

  emb = df_display["embeddedness_score"].to_numpy()
  emb_min, emb_max = float(np.min(emb)), float(np.max(emb))
  denom = (emb_max - emb_min) if emb_max > emb_min else 1.0


  for n in G.nodes():
    x, y = pos[n]
    node_x.append(x); node_y.append(y)

    role = df_display.loc[n,"role_name"]
    conf = df_display.loc[n, "confidence"]
    node_color.append(ROLE_COLORS.get(role, "#95a5a6"))

    s = 12 + 18 * ((df_display.loc[n,"embeddedness_score"] - emb_min) / denom)
    sizes.append(s)

    hover.append(
      f"Member {n}"
      f"<br>Role: {role}"
      f"<br>Confidence: {conf:.0%}"

    )

  fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode="markers",
    marker=dict(size=sizes, color=node_color, line=dict(width=1,color="rgba(20,20,20,0.6)")),
    hovertext=hover,
    hoverinfo="text"
  ))

  fig.update_layout(
    margin=dict(l=0,r=0,t=0,b=0),
    showlegend=False,
    height=600
  )
  return fig

col1, col2 = st.columns([2.2, 1])
with col1:
  st.subheader("Role Map")
  st.plotly_chart(plot_network(G, df_display), use_container_width=True)
with col2:
  st.subheader("Member inspection")

  node_id = st.selectbox("Select a member", df_display["node"].tolist(), index=0)
  row = df_display.loc[node_id]

  st.markdown(f"**Member:** {node_id}")
  st.markdown(f"**Assigned role:** {row['role_name']}")
  st.progress(float(row["confidence"]))
  st.caption(f"Confidence: {row['confidence']:.0%} (agreement acroos methods)")
  if row["confidence"] < 0.5:
    st.warning("Low agreement across methods. Treat this role as uncertain.")
  
  st.caption(role_explanation(row["role_name"])) 

  st.markdown("---")
  st.markdown("**What this means**")
  st.write(role_explanation(row["role_name"]))

  st.markdown("**Why we think so**")
  for bullet in why_we_think_so(row["role_name"]):
    st.write("• " + bullet)

  with st.expander("Show technical details"):
    st.write(
        f"Overall involvement in the network: **{row['embeddedness_score']:.2f}** "
        "(how strongly this member is connected through direct and indirect links)"
    )

    st.write(
        f"How often others reach this member: **{row['in_total']:.2f}** "
        "(how much activity or attention flows *towards* them)"
    )

    st.write(
        f"How often this member reaches others: **{row['out_total']:.2f}** "
        "(how much activity or influence flows *from* them)"
    )

    st.write(
        f"Balance of reaching out vs being reached: **{row['net_flow']:.2f}** "
        "(positive = more outgoing, negative = more incoming)"
    )


st.subheader("Members to review")

def review_reason(role, conf, emb, emb_thr):
    if conf < 0.5:
        return "Uncertain role (methods disagree)"
    if "Core-like" in role:
        return "Highly central (high impact)"
    if emb >= emb_thr:
        return "Very embedded (many connections)"
    return "Routine"

emb_thr = df_display["embeddedness_score"].quantile(0.90)

short = df_display.copy()
short["reason"] = [
    review_reason(r, c, e, emb_thr)
    for r, c, e in zip(short["role_name"], short["confidence"], short["embeddedness_score"])
]

# prioritize: uncertain first, then core-like, then very embedded
priority_order = {"Uncertain role (methods disagree)": 0,
                  "Highly central (high impact)": 1,
                  "Very embedded (many connections)": 2,
                  "Routine": 3}

short["priority"] = short["reason"].map(priority_order).fillna(9)

shortlist = short.sort_values(["priority", "embeddedness_score"], ascending=[True, False]).head(20)

st.dataframe(
    shortlist[["node", "role_name", "confidence", "reason"]],
    use_container_width=True,
    hide_index=True
)

  

  


                                    
               
              
