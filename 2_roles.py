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

st.title("ROLE IDENTIFICATION")
st.caption("SOCIAL ROLE ANALYSIS ENGINE")

with st.expander("📘Quick guide", expanded=True):
    st.markdown(
        """
        This page supports **operational role assignment** based on network structure.

        **Workflow:**
        1. Select a **role identification method** from the left panel.
        2. Inspect the **colored network map** to understand role distribution.
        3. Select a **member** on the right to view role rationale, confidence, and evidence.
        """
    )





@st.cache_data
def load_adjacency():
  A = scipy.io.mmread("data/clandestine_network_example.mtx").tocsr()
  return A

A = load_adjacency()
G = nx.from_scipy_sparse_array(A)

def get_direct_contacts(G: nx.Graph, node_id: int) -> list[int]:
    """Sorted list of direct neighbors (immediate contacts) for a node."""
    return sorted(list(G.neighbors(int(node_id))))


@st.cache_data
def compute_roles(_A):
  return run_all_role_methods(_A)

results = compute_roles(A)

df_flow = results["df_flow"]
df_dist = results["df_dist"]
df_cent = results["df_cent"]
df_overlap = results["df_overlap"]

df_flow = df_flow.set_index("node", drop=False)
df_dist = df_dist.set_index("node", drop=False)
df_cent = df_cent.set_index("node", drop=False)
df_overlap = df_overlap.set_index("node", drop=False)

# Method selector
with st.sidebar:
    st.subheader("Roles settings")
    method_ui = st.selectbox(
        "Role method",
        ["Influence (flow)", "Core distance", "Importance type", "Similar contacts"],
        index=None,
        placeholder="Select a method...",
        help=(
            "Each method assigns roles using a different network signal. "
            "Confidence shows how much the other methods agree."
        )
    )


if method_ui is None:
    st.warning("Select a role method to generate the role map and member inspection.")
    st.stop()

METHOD_KEY = {
    "Influence (flow)": "Flow",
    "Core distance": "Distance",
    "Importance type": "Centrality",
    "Similar contacts": "Overlap"
}[method_ui]

st.caption(f"Selected role method: **{method_ui}**")


if METHOD_KEY == "Flow":
    st.markdown("**What this method measures:** Influence via multi-step information flow through the network.")
elif METHOD_KEY == "Distance":
    st.markdown("**What this method measures:** How many steps away a member is from the high-degree core.")
elif METHOD_KEY == "Centrality":
    st.markdown("**What this method measures:** Whether a member behaves like a hub, bridge, or peripheral based on centrality signals.")
else:
    st.markdown("**What this method measures:** Similarity of contact patterns (structural equivalence / overlap).")



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
  ref = votes[METHOD_KEY]          # selected method becomes the reference
  agree = sum(v == ref for v in votes.values())
  return agree / len(votes)

def normalize_role_label(method_key: str, i: int) -> str:
    """Return a role label in the SAME 4 categories, regardless of method."""
    votes = method_vote_for_node(i)

    if method_key == "Flow":
        return votes["Flow"]

    if method_key == "Distance":
        dr = df_dist.loc[i, "distance_role"]
        if dr == "High-degree Core":
            return "Core-like (high embeddedness)"
        elif dr == "Near high-degree core":
            return "Intermediate (moderate embeddedness)"
        elif dr == "Isolated":
            return "Extreme peripheral / near isolated"
        else:
            return "Peripheral (low embeddedness)"

    if method_key == "Centrality":
        cr = df_cent.loc[i, "centrality_role_name"]
        if cr in ["Hub-like", "Influential"]:
            return "Core-like (high embeddedness)"
        elif cr == "Bridge-like":
            return "Intermediate (moderate embeddedness)"
        else:
            return "Peripheral (low embeddedness)"

    # Overlap
    deg = df_overlap.loc[i, "degree"]
    if deg <= 1:
        return "Extreme peripheral / near isolated"
    elif deg <= np.median(df_overlap["degree"]):
        return "Peripheral (low embeddedness)"
    else:
        return "Intermediate (moderate embeddedness)"


# Display dataframe
df_display = df_flow[["node","embeddedness_score","in_total","out_total","net_flow"]].copy()
df_display = df_display.set_index("node", drop=False)

# role label depends on selected method
df_display["role_label"] = [normalize_role_label(METHOD_KEY, i) for i in df_display["node"]]

# confidence depends on selected method
df_display["confidence"] = [confidence_for_node(i) for i in df_display["node"]]



ROLE_COLORS = {
  "Core-like (high embeddedness)": "#e74c3c",
  "Intermediate (moderate embeddedness)": "#f39c12",
  "Peripheral (low embeddedness)": "#3498db",
  "Extreme peripheral / near isolated": "#7f8c8d"
}

def role_explanation(role):
  if "Extreme peripheral" in role:
    return "Very few connections; likely isolated or inactive."
  if "Core-like" in role:
    return "Highly connected and influential across multiple parts of the network."
  if "Intermediate" in role:
    return "Moderately connected; often acts as a link between core and peripheral members."
  if "Peripheral" in role:
    return "Limited interactions; participates in few network pathways."
  return "Role description not available."



def why_we_think_so(role: str) -> list[str]:
    if "Extreme peripheral" in role:
        return [
            "Has very few (or no) observable ties in the data.",
            "Minimal network presence detected."
        ]
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
    return ["No explanation available."]



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

    role = df_display.loc[n,"role_label"]
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


st.caption(
    "Summary for the selected method: counts show how many members fall in each role; "
    "Avg. confidence = average % of methods that assign the same role as the selected method."
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Core-like", int((df_display["role_label"] == "Core-like (high embeddedness)").sum()))
m2.metric("Intermediate", int((df_display["role_label"] == "Intermediate (moderate embeddedness)").sum()))
m3.metric("Peripheral", int((df_display["role_label"] == "Peripheral (low embeddedness)").sum()))
m4.metric("Extreme peripheral", int((df_display["role_label"] == "Extreme peripheral / near isolated").sum()))
m5.metric("Avg. confidence", f"{df_display['confidence'].mean():.0%}")


col1, col2 = st.columns([3, 2])
with col1:
  st.subheader("Role Map")
  st.info(
  "How to read: Colors = role type. Bigger nodes = more embedded in the network. "
  "Confidence = how much the different methods agree."
  )  
  st.caption(
    "Legend: Core-like = red, Intermediate = orange, Peripheral = blue, Extreme peripheral = gray."
  )

  st.plotly_chart(plot_network(G, df_display), use_container_width=True)
with col2:
  st.subheader("Member inspection")

  node_id = st.selectbox("Select a member", df_display["node"].tolist(), index=0)
  row = df_display.loc[node_id]
  contacts = get_direct_contacts(G, int(node_id))


  st.caption("Need context? Open **Direct contacts** to see who this member is directly linked to.")
  with st.popover(f"Direct contacts ({len(contacts)})"):
      if contacts:
          st.write(", ".join(map(str, contacts)))
      else:
          st.write("No direct contacts in the data.")

  st.markdown(f"**Member:** {node_id}")
  st.markdown(f"**Assigned role:** {row['role_label']}")
  st.progress(float(row["confidence"]))
  st.caption(f"Confidence: {row['confidence']:.0%} (agreement vs other methods)")
  if row["confidence"] < 0.5:
    st.warning("Low agreement across methods. Treat this role as uncertain.")

  st.write(role_explanation(row["role_label"]))
  for bullet in why_we_think_so(row["role_label"]):
    st.write("• " + bullet)

  st.caption("These are relative network indicators (use for comparison, not absolute interpretation).")

  with st.expander("Evidence (advanced)"):
    st.write(f"Selected method: **{method_ui}**")
    
    if METHOD_KEY == "Flow":
    
      emb = float(row["embeddedness_score"])
      outv = float(row["out_total"])
      inv = float(row["in_total"])

      emb_q50, emb_q75 = np.quantile(df_display["embeddedness_score"], [0.50, 0.75])
      out_q50, out_q75 = np.quantile(df_display["out_total"], [0.50, 0.75])
      in_q50,  in_q75  = np.quantile(df_display["in_total"],  [0.50, 0.75])

      def level(x, q50, q75):
          return "High" if x >= q75 else "Medium" if x >= q50 else "Low"

      st.write(f"Overall involvement (relative): **{level(emb, emb_q50, emb_q75)}**")
      st.write(f"Reaches others (relative): **{level(outv, out_q50, out_q75)}**")
      st.write(f"Is reached by others (relative): **{level(inv, in_q50, in_q75)}**")

      with st.expander("Show exact flow values (technical)"):
          st.write(f"Embeddedness score: {emb:.2f}")
          st.write(f"Out total: {outv:.2f}")
          st.write(f"In total: {inv:.2f}")


    elif METHOD_KEY == "Distance":
      d = df_dist.loc[node_id, "dist_to_core"]
      st.write(f"Distance to core: **{d if np.isfinite(d) else '∞'}** (steps away from high-degree core)")
    elif METHOD_KEY == "Centrality":
      # Plain-language evidence (no jargon)
      deg = float(df_cent.loc[node_id, "degree"])
      btw = float(df_cent.loc[node_id, "betweenness"])
      eig = float(df_cent.loc[node_id, "eigenvector"])
      katz = float(df_cent.loc[node_id, "katz"])
      st.write(
          f"Connector tendency (bridges groups): **{'High' if btw >= np.quantile(df_cent['betweenness'], 0.75) else 'Medium' if btw >= np.quantile(df_cent['betweenness'], 0.50) else 'Low'}**"
      )

      st.write(
          f"Influence signal (connected to important members): **{'High' if (eig + katz) >= np.quantile((df_cent['eigenvector'] + df_cent['katz']), 0.75) else 'Medium' if (eig + katz) >= np.quantile((df_cent['eigenvector'] + df_cent['katz']), 0.50) else 'Low'}**"
      )


      with st.expander("Show centrality metrics (technical)"):
          st.write(f"Betweenness: {btw:.4f}")
          st.write(f"Eigenvector: {eig:.4f}")
          st.write(f"Katz: {katz:.4f}")


    elif METHOD_KEY == "Overlap":
        deg = float(df_overlap.loc[node_id, "degree"])

        # similarity proxy: degree relative to others
        q50, q75 = np.quantile(df_overlap["degree"], [0.50, 0.75])
        similarity_level = (
            "High" if deg >= q75 else
            "Medium" if deg >= q50 else
            "Low"
        )

        st.write(f"Contact similarity (how typical this member's contacts are): **{similarity_level}** "
        )

    



   



  

  


                                    
               
              
