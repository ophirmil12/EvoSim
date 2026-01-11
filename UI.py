import streamlit as st
import yaml
import pandas as pd
import numpy as np
import os
import sys

# ==========================================
# 0. Page Configuration (Wide & Visual Tweaks)
# ==========================================
st.set_page_config(
    page_title="SERVINE Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for centering images and adding style
st.markdown("""
    <style>
        /* Center images and give them a nice shadow */
        .stImage {
            display: flex;
            justify_content: center;
        }
        .stImage > img {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            max-width: 90% !important; /* Keep it large but with margins */
        }
        /* Make headers centered */
        h1, h2, h3 {
            text-align: center;
        }
        /* Improve tab font size */
        button[data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 1. Helper Functions
# ==========================================

def get_default_epoch():
    return {
        "name": "New_Epoch",
        "generations": 500,
        "mutator": {"rate": 0.001, "transition_bias": 1.0},
        "fitness": {"type": "purifying", "params": {"intensity": 0.05}}
    }


def get_default_sampling():
    return {
        "type": "diversity",
        "interval": 10,
        "params": {}
    }


# ==========================================
# 2. Session State Initialization
# ==========================================

if "config" not in st.session_state:
    st.session_state.config = {
        "genome": {"length": 50},
        "population": {
            "initial_size": 200,
            "initial_sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        },
        "epochs": [get_default_epoch()],
        "sampling": [get_default_sampling()]
    }

config = st.session_state.config

# ==========================================
# 3. Sidebar: Global Settings
# ==========================================

st.sidebar.title("🧬 Simulation Config")
st.sidebar.markdown("---")

st.sidebar.header("1. Genome")
config["genome"]["length"] = st.sidebar.number_input(
    "Genome Length", min_value=1, value=config["genome"]["length"]
)

st.sidebar.header("2. Population")
config["population"]["initial_size"] = st.sidebar.number_input(
    "Initial Population Size", min_value=1, value=config["population"]["initial_size"]
)

current_seq = config["population"].get("initial_sequence", "")
config["population"]["initial_sequence"] = st.sidebar.text_area(
    "Initial Sequence (Optional)", value=current_seq, height=100
)

# ==========================================
# 4. Epochs Configuration
# ==========================================

st.title("Evolutionary Simulation Controller")
st.markdown("---")
st.header("3. Epochs Definition")

indices_to_remove = []

for i, epoch in enumerate(config["epochs"]):
    with st.expander(f"Epoch {i + 1}: {epoch.get('name', 'Unnamed')}", expanded=False):
        c1, c2 = st.columns([2, 1])
        epoch["name"] = c1.text_input("Epoch Name", value=epoch["name"], key=f"ep_name_{i}")
        epoch["generations"] = c2.number_input("Generations", min_value=1, value=epoch["generations"],
                                               key=f"ep_gen_{i}")

        st.markdown("#### Mutator Settings")
        m1, m2 = st.columns(2)
        epoch["mutator"]["rate"] = m1.number_input(
            "Mutation Rate", step=0.0001, format="%.5f",
            value=float(epoch["mutator"].get("rate", 0.001)), key=f"mut_rate_{i}"
        )
        epoch["mutator"]["transition_bias"] = m2.number_input(
            "Transition Bias", value=float(epoch["mutator"].get("transition_bias", 1.0)), key=f"mut_bias_{i}"
        )

        st.markdown("#### Fitness Landscape")
        f1, f2 = st.columns(2)
        fitness_options = ["purifying", "exposure", "frequency", "neutral", "epistatic"]
        current_type = epoch["fitness"].get("type", "purifying")
        if current_type not in fitness_options: fitness_options.append(current_type)

        new_type = f1.selectbox("Fitness Type", fitness_options, index=fitness_options.index(current_type),
                                key=f"fit_type_{i}")
        epoch["fitness"]["type"] = new_type

        params = epoch["fitness"].get("params", {})
        if new_type == "purifying":
            params["intensity"] = f2.number_input("Intensity", value=float(params.get("intensity", 0.05)),
                                                  key=f"fit_p_{i}")
        elif new_type == "exposure":
            params["intensity"] = f2.number_input("Intensity", value=float(params.get("intensity", 0.15)),
                                                  key=f"fit_p_int_{i}")
            params["update_interval"] = st.number_input("Update Interval",
                                                        value=int(params.get("update_interval", 100)),
                                                        key=f"fit_p_upd_{i}")
        elif new_type == "frequency":
            params["pressure"] = f2.number_input("Pressure", value=float(params.get("pressure", 1.5)),
                                                 key=f"fit_p_pres_{i}")

        epoch["fitness"]["params"] = params

        if st.button("🗑️ Remove Epoch", key=f"del_ep_{i}"):
            indices_to_remove.append(i)

if indices_to_remove:
    for index in sorted(indices_to_remove, reverse=True):
        del config["epochs"][index]
    st.rerun()

if st.button("➕ Add New Epoch"):
    config["epochs"].append(get_default_epoch())
    st.rerun()

# ==========================================
# 5. Sampling Configuration
# ==========================================

st.divider()
st.header("4. Sampling & Outputs")

sampling_remove_indices = []

for j, sample in enumerate(config["sampling"]):
    cols = st.columns([2, 1, 2, 0.5])

    sample_types = ["diversity", "tree", "pairwise", "haplotype", "fitness", "fasta", "identity"]
    current_samp_type = sample.get("type", "diversity")
    if current_samp_type not in sample_types: current_samp_type = "diversity"

    sample["type"] = cols[0].selectbox("Type", sample_types, index=sample_types.index(current_samp_type),
                                       key=f"samp_type_{j}", label_visibility="collapsed")
    sample["interval"] = cols[1].number_input("Interval", min_value=1, value=int(sample.get("interval", 10)),
                                              key=f"samp_int_{j}", label_visibility="collapsed")
    sample["file"] = f"output/{sample['type']}.csv"

    if sample["type"] == "haplotype":
        current_params = sample.get("params", {})
        top_n = cols[2].number_input("Top N", min_value=1, value=int(current_params.get("top_n", 10)),
                                     key=f"samp_p_{j}")
        sample["params"] = {"top_n": top_n}
    else:
        cols[2].write("")

    if cols[3].button("x", key=f"rm_samp_{j}"):
        sampling_remove_indices.append(j)

if sampling_remove_indices:
    for index in sorted(sampling_remove_indices, reverse=True):
        del config["sampling"][index]
    st.rerun()

if st.button("➕ Add Output"):
    config["sampling"].append(get_default_sampling())
    st.rerun()

# ==========================================
# 6. Export, Run & Visualization
# ==========================================

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path: sys.path.append(base_dir)
src_path = os.path.join(base_dir, 'src')
if src_path not in sys.path: sys.path.append(src_path)

simulation_function = None
import_error_message = ""

try:
    from src.servine.cli import run_simulation_from_config

    simulation_function = run_simulation_from_config
except ImportError as e1:
    try:
        from servine.cli import run_simulation_from_config

        simulation_function = run_simulation_from_config
    except ImportError as e2:
        import_error_message = f"Attempt 1 failed: {e1}\nAttempt 2 failed: {e2}"

if simulation_function is None:
    st.divider()
    st.error("❌ Critical Error: Could not load simulation engine.")
    st.code(import_error_message)
    st.stop()

st.divider()
st.subheader("Actions")

col_actions1, col_actions2 = st.columns(2)

with col_actions1:
    st.download_button(
        label="💾 Download Config YAML",
        data=yaml.dump(config, sort_keys=False),
        file_name="config.yaml",
        mime="text/yaml"
    )

with col_actions2:
    if st.button("Run Simulation!", type="primary"):
        with st.status("Running Simulation...", expanded=True) as status:
            st.write("Initializing simulation engine...")
            try:
                simulation_function(config)
                st.write("Finalizing analysis...")
                status.update(label="Simulation Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Simulation Failed: {e}")
                status.update(label="Error Occurred", state="error")
                st.stop()

# ==========================================
# 7. BEAUTIFUL VISUALIZATION (TABS)
# ==========================================
st.divider()
st.header("Simulation Results")

tab_plots, tab_trees, tab_data = st.tabs([
    "Population Dynamics",
    "Phylogeny & Evolution",
    "Raw Data"
])

# --- TAB 1: DYNAMICS (Plots) ---
with tab_plots:
    st.caption("Visualizing population statistics over time.")
    has_plots = False

    for sample in config["sampling"]:
        sample_type = sample.get("type")
        csv_path = sample.get("file")
        if not csv_path or sample_type == "tree" or sample_type == "fasta":
            continue  # Skip trees and fasta here

        png_path = csv_path.replace(".csv", ".png")
        if os.path.exists(png_path):
            has_plots = True
            st.subheader(f"{sample_type.replace('_', ' ').title()}")
            st.image(png_path, use_container_width=True)
            st.divider()

    if not has_plots:
        st.info("No dynamics plots found. Try adding 'diversity', 'fitness', or 'haplotype' outputs.")

# --- TAB 2: TREES ---
with tab_trees:
    st.caption("Evolutionary history and lineages.")
    has_trees = False

    for sample in config["sampling"]:
        if sample.get("type") == "tree":
            csv_path = sample.get("file")
            ltt_plot = csv_path.replace(".csv", ".png")
            topo_plot = csv_path.replace(".csv", "_tree.png")

            if os.path.exists(topo_plot):
                has_trees = True
                st.subheader("Phylogenetic Tree (Topology)")
                # Centered, large image
                st.image(topo_plot, use_container_width=True)

            if os.path.exists(ltt_plot):
                has_trees = True
                st.divider()
                st.subheader("Lineages Through Time (LTT)")
                st.image(ltt_plot, use_container_width=True)

    if not has_trees:
        st.info("No phylogenetic trees generated. Add a 'tree' output to see results here.")

# --- TAB 3: RAW DATA ---
with tab_data:
    st.caption("Access the raw CSV and FASTA files generated by the simulation.")

    for sample in config["sampling"]:
        sample_type = sample.get("type")
        csv_path = sample.get("file")

        if not csv_path or not os.path.exists(csv_path):
            continue

        with st.expander(f"📄 {sample_type.upper()} Data ({csv_path})"):
            if sample_type == "fasta":
                with open(csv_path, "r") as f:
                    st.code(f.read(2000) + "\n... (truncated)", language="text")
            else:
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)