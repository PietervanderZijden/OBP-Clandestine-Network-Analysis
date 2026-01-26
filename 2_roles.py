import streamlit as st
import pandas as pd
import numpy as np
import scipy.io
import networkx as nx
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config

from ui_components import apply_tactical_theme, COLOR_VOID, COLOR_WIRE, COLOR_STEEL, COLOR_ALERT
from roles_logic import run_all_role_methods
from collections import Counter


# --- NEW IMPORT ---
from src.data_manager import get_active_network

st.set_page_config(layout="wide")
apply_tactical_theme()

st.title("Role Identification")

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

# --- CHANGED: DATA LOADING SECTION ---
# We use the manager to get the G object directly
G, metadata = get_active_network()

@st.cache_data
def compute_layout(_graph, source_name):
    pos = nx.spring_layout(_graph, seed=42)
    return {int(u): (pos[u][0] * 1000, pos[u][1] * 1000) for u in _graph.nodes()}

layout_map = compute_layout(G, metadata["name"])


# The roles_logic.py script expects a SciPy Sparse Matrix (A),
# so we simply convert the NetworkX graph 'G' into that format.
nodes = sorted(G.nodes())
A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr")


# -------------------------------------

def get_direct_contacts(G: nx.Graph, node_id: int) -> list[int]:
    """Sorted list of direct neighbors (immediate contacts) for a node."""
    return sorted(list(G.neighbors(int(node_id))))


@st.cache_data
def compute_roles_for_source(_A, source_name: str):
    return run_all_role_methods(_A)

results = compute_roles_for_source(A, metadata["name"])


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
    target_name = metadata.get("name", "Unknown")
    st.subheader("Roles settings")
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin: 4px 0 6px 0;">
          <div style="font-size:12px; color:#8b949e;">Select Method</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # dataset line
    st.markdown(
        f"""
        <div style="font-family:'Share Tech Mono', monospace; font-size:12px; color:#8b949e; margin-bottom:10px;">
            dataset: <span style="color:var(--color-accent)">{target_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    
    method_ui = st.radio(
        label="",
        options=[
            "Influence (flow)",
            "Core distance",
            "Importance type",
            "Similar contacts",
        ],
        index=0,
        label_visibility="collapsed",
    )

    # --- Description directly under selection ---
    desc = {
        "Influence (flow)": "Measures influence via multi-step information flow through the network.",
        "Core distance": "Measures how far each member is from the highly connected core.",
        "Importance type": "Identifies hubs, bridges, and peripheral members using centrality signals.",
        "Similar contacts": "Groups members by similarity of their contact patterns.",
    }
    st.caption(desc[method_ui])

    st.markdown(
        "<div style='height:1px; background: rgba(255,255,255,0.12); margin-top:12px;'></div>",
        unsafe_allow_html=True,
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


def consensus_for_node(i: int) -> tuple[str, float, dict[str, int]]:
    votes = method_vote_for_node(i)              # dict: method -> role
    counts = Counter(votes.values())             # role -> count
    consensus_role, top_count = counts.most_common(1)[0]
    consensus_strength = top_count / len(votes)
    return consensus_role, consensus_strength, dict(counts)

def votes_summary_for_node(i: int):
    votes = method_vote_for_node(i)
    counts = Counter(votes.values())
    ranked = counts.most_common()

    consensus_role, top_count = ranked[0]
    consensus_strength = top_count / len(votes)

    runner_up_role, runner_up_strength = None, None
    if len(ranked) > 1:
        runner_up_role, runner_up_count = ranked[1]
        runner_up_strength = runner_up_count / len(votes)

    return votes, counts, consensus_role, consensus_strength, runner_up_role, runner_up_strength


# Display dataframe
df_display = df_flow[["node","embeddedness_score","in_total","out_total","net_flow"]].copy()
df_display = df_display.set_index("node", drop=False)

# role label depends on selected method
df_display["role_label"] = [normalize_role_label(METHOD_KEY, i) for i in df_display["node"]]

# confidence depends on selected method
df_display["confidence"] = [confidence_for_node(i) for i in df_display["node"]]
df_display["consensus_role"] = [consensus_for_node(i)[0] for i in df_display["node"]]
df_display["consensus_strength"] = [consensus_for_node(i)[1] for i in df_display["node"]]




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



def agraph_network(G: nx.Graph, df_display: pd.DataFrame):
    pos = nx.spring_layout(G, seed=42)

    nodes = []
    for n in G.nodes():
        if n not in df_display.index:
            continue

        role = df_display.loc[n, "role_label"]
        conf = float(df_display.loc[n, "confidence"])
        emb = float(df_display.loc[n, "embeddedness_score"])
        cons_role = df_display.loc[n, "consensus_role"]
        cons_strength = df_display.loc[n, "consensus_strength"]

        nodes.append(
            Node(
                id=str(n),
                label=str(n),
                size=12 + 18 * (
                    (emb - df_display["embeddedness_score"].min()) /
                    (df_display["embeddedness_score"].max()
                     - df_display["embeddedness_score"].min() + 1e-9)
                ),
                color=ROLE_COLORS.get(role, "#95a5a6"),
                title=(
                    f"Member {n}\n"
                    f"Selected role: {role}\n"
                    f"Agreement: {conf:.0%}\n"
                    f"Majority role: {cons_role}\n"
                    f"Majority strength: {cons_strength:.0%}"
                ),
                x=float(pos[n][0]) * 1000,
                y=float(pos[n][1]) * 1000,
            )
        )


            

    # Build edges
    edges = []
    for u, v in G.edges():
        edges.append(
            Edge(
                source=str(int(u)),
                target=str(int(v)),
                color=COLOR_WIRE,
                width=1,
                opacity=0.35,
                type="STRAIGHT",
            )
        )


    
    config = Config(
        width="100%",
        height=600,
        directed=False,
        physics=False,
        staticGraph=True,
        nodeHighlightBehavior=True,
        backgroundColor="#000000",  # or use their theme_style() logic
        visjs_config={"interaction": {"hover": True}},
    )


    
    return agraph(nodes=nodes, edges=edges, config=config)



st.caption(
    "Summary: role counts are based on the selected method. "
    "Avg agreement = mean agreement with the selected method. "
    "Avg consensus = mean majority strength across all methods."
)


m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Core-like", int((df_display["role_label"] == "Core-like (high embeddedness)").sum()))
m2.metric("Intermediate", int((df_display["role_label"] == "Intermediate (moderate embeddedness)").sum()))
m3.metric("Peripheral", int((df_display["role_label"] == "Peripheral (low embeddedness)").sum()))
m4.metric("Extreme peripheral", int((df_display["role_label"] == "Extreme peripheral / near isolated").sum()))
m5.metric("Avg agreement", f"{df_display['confidence'].mean():.0%}")
m6.metric("Avg majority agreement", f"{df_display['consensus_strength'].mean():.0%}")


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

  selected = agraph_network(G, df_display)



with col2:
  st.subheader("Member inspection")

  # If user clicked a node in the graph, use it; otherwise fallback to selectbox
  # Ensure we have a valid default even if the graph selection is empty
  default_node = int(df_display["node"].iloc[0])
  if selected:
      try:
          default_node = int(selected)
      except:
          pass

  # Filter list to valid nodes
  valid_nodes = df_display["node"].tolist()
  if default_node not in valid_nodes:
      default_node = valid_nodes[0]

  node_id = st.selectbox(
      "Select a member",
      valid_nodes,
      index=valid_nodes.index(default_node)
  )

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
  # --- Agreement (selected method) + Consensus (majority) + "between roles" ---
  votes, counts, consensus_role, consensus_strength, runner_up_role, runner_up_strength = votes_summary_for_node(int(node_id))

  agreement_selected = float(row["confidence"])
  n_methods = len(votes)

  st.progress(agreement_selected)
  st.caption(
      f"Agreement with selected method: {agreement_selected:.0%}",
      help=(
          "Fraction of methods that assign the SAME role as the chosen method. "
          "Example: 1 out of 4 → 25%."
      )
  )

  st.caption(
      f"Majority: **{consensus_role}** ({consensus_strength:.0%})",
      help=(
          "Majority: most common role across all methods. "
          "Strength: fraction of methods voting for that role."
      )
  )

    # Warnings (two different situations)
  if consensus_strength <= 0.5:
        st.warning("Low overall agreement. Multiple role interpretations are plausible")
  elif agreement_selected <= 0.5 and row["role_label"] != consensus_role:
        st.warning("Selected method disagrees with the majority. Review alternative roles below.")

    # Show the alternatives when it's not unanimous OR when selected method disagrees
  if consensus_strength < 1.0 or agreement_selected <= 0.5:
        with st.expander("Roles suggested by each method", expanded=(agreement_selected <= 0.5)):
            for m in ["Flow", "Distance", "Centrality", "Overlap"]:
                st.write(f"• **{m}**: {votes[m]}")

            

    
  st.write(role_explanation(row["role_label"]))
  for bullet in why_we_think_so(row["role_label"]):
    st.write("• " + bullet)

  st.caption("These are relative network indicators (use for comparison, not absolute interpretation).")

  with st.expander("Evidence (advanced)"):
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
