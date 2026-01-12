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

# Custom CSS
st.markdown("""
    <style>
        .stImage {
            display: flex;
            justify_content: center;
        }
        .stImage > img {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        }

        h1, h2, h3 {
            font-family: 'Helvetica', sans-serif;
            text-align: left;
        }

        button[data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 600;
        }

        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER IMAGE & TITLE LOGIC ---
banner_path = "picture.png"

if os.path.exists(banner_path):
    col_logo, col_title = st.columns([1, 7])

    with col_logo:
        st.image(banner_path, width=130)

    with col_title:
        # We use HTML here to force a larger font size (e.g. 70px)
        st.markdown("""
            <h1 style='font-size: 70px; margin-bottom: 0; padding-top: 0;'>
                SERVINE
            </h1>
        """, unsafe_allow_html=True)
        st.caption("Evolutionary Simulation of Viral Populations")

else:
    st.title("SERVINE Simulator")

# ==========================================
# 1. Helper Functions
# ==========================================

def get_default_epoch():
    return {
        "name": "New_Epoch",
        "generations": 500,
        "mutator": {
            "type": "nucleotide",
            # Now 'rate' and 'transition_bias' sit inside params!
            "params": {
                "rate": 0.001,
                "transition_bias": 2.0
            }
        },
        "fitness": {
            "type": "purifying",
            "params": {"intensity": 0.05}
        }
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

# --- 1. Genome Section ---
st.sidebar.header("1. Genome")
config["genome"]["length"] = st.sidebar.number_input(
    "Genome Length",
    min_value=1,
    value=config["genome"]["length"],
    help="Total number of nucleotides."
)

# --- 2. Population Section ---
st.sidebar.header("2. Population")

# A. Population Size
config["population"]["initial_size"] = st.sidebar.number_input(
    "Total Population Size",
    min_value=1,
    value=config["population"]["initial_size"]
)

# B. Initialization Mode Selection
# We define constants to ensure the text matches exactly in the if-statement
MODE_SINGLE = "Single Sequence"
MODE_DIST = "Distribution"

# Determine default index based on existing config
default_idx = 1 if "initial_distribution" in config["population"] else 0

init_mode = st.sidebar.radio(
    "Initialization Mode",
    options=[MODE_SINGLE, MODE_DIST],
    index=default_idx
)

if init_mode == MODE_SINGLE:
    # --- Mode 1: Single Sequence ---

    # 1. Clean up conflicting key if exists
    if "initial_distribution" in config["population"]:
        del config["population"]["initial_distribution"]

    # 2. Ensure initial_sequence key exists and is a string
    current_seq = config["population"].get("initial_sequence", "")
    if not isinstance(current_seq, str):
        current_seq = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

    # 3. Display Text Area
    new_seq = st.sidebar.text_area(
        "Initial Sequence",
        value=current_seq,
        height=100,
        placeholder="e.g., ACGTACGT..."
    )

    # 4. Save to config
    config["population"]["initial_sequence"] = new_seq.strip()

else:
    # --- Mode 2: Custom Distribution ---

    # 1. Clean up conflicting key if exists
    if "initial_sequence" in config["population"]:
        del config["population"]["initial_sequence"]

    st.sidebar.info("Format: SEQUENCE : COUNT")

    # 2. Prepare default text for the text area (Convert Dict -> String)
    current_dist = config["population"].get("initial_distribution", {"AAAAA": 10, "TTTTT": 10})
    dist_text = ""
    if isinstance(current_dist, dict):
        for seq, count in current_dist.items():
            dist_text += f"{seq}: {count}\n"

    # 3. Display Text Area
    raw_dist = st.sidebar.text_area(
        "Distribution Config",
        value=dist_text,
        height=150,
        placeholder="AAAAA: 50\nCCCCC: 50"
    )

    # 4. Parse the text area back into a dictionary
    parsed_dist = {}
    total_count = 0

    for line in raw_dist.split("\n"):
        if ":" in line:
            parts = line.split(":")
            seq_key = parts[0].strip().upper()
            try:
                cnt_val = int(parts[1].strip())
                parsed_dist[seq_key] = cnt_val
                total_count += cnt_val
            except ValueError:
                pass

    config["population"]["initial_distribution"] = parsed_dist

    # 5. Validation Feedback
    expected_size = config['population']['initial_size']
    if total_count != expected_size:
        st.sidebar.warning(
            f"Sum ({total_count}) != Total Size ({expected_size}).\n"
            "The registry will adjust automatically."
        )
    else:
        st.sidebar.success(f"Matches Total Size: {total_count}")

# ==========================================
# 4. Epochs Configuration
# ==========================================

st.markdown("---")
st.header("3. Epochs Definition")

indices_to_remove = []

for i, epoch in enumerate(config["epochs"]):
    with st.expander(f"Epoch {i + 1}: {epoch.get('name', 'Unnamed')}", expanded=False):

        # --- General Epoch Settings ---
        c1, c2 = st.columns([2, 1])
        epoch["name"] = c1.text_input("Epoch Name", value=epoch["name"], key=f"ep_name_{i}")
        epoch["generations"] = c2.number_input("Generations", min_value=1, value=epoch["generations"],
                                               key=f"ep_gen_{i}")

        st.divider()

        # --- MUTATOR SETTINGS ---
        st.markdown("#### 🧬 Mutation Model")

        if "params" not in epoch["mutator"]:
            epoch["mutator"]["params"] = {}

        mut_params = epoch["mutator"]["params"]

        # 1. Mutator Type Selection
        mutator_types = ["nucleotide", "unify", "hotcold"]
        current_mut_type = epoch["mutator"].get("type", "nucleotide")
        if current_mut_type not in mutator_types: current_mut_type = "nucleotide"

        epoch["mutator"]["type"] = st.selectbox(
            "Mutator Type",
            mutator_types,
            index=mutator_types.index(current_mut_type),
            key=f"mut_type_{i}"
        )

        # 2. Base Parameters
        m1, m2 = st.columns(2)

        current_rate = float(mut_params.get("rate", 0.001))
        mut_params["rate"] = m1.number_input(
            "Base Mutation Rate", step=0.0001, format="%.5f",
            value=current_rate, key=f"mut_rate_{i}"
        )

        # 3. Model-Specific Parameters & Cleanup
        if epoch["mutator"]["type"] == "nucleotide":
            current_bias = float(mut_params.get("transition_bias", 2.0))
            mut_params["transition_bias"] = m2.number_input(
                "Transition Bias (Ts/Tv)",
                value=current_bias,
                key=f"mut_bias_{i}"
            )
            for key in ["variable_kmers", "conserved_kmers", "k_high", "k_low", "threshold"]:
                mut_params.pop(key, None)

        elif epoch["mutator"]["type"] == "unify":
            m2.write("")
            keys_to_remove = ["transition_bias", "variable_kmers", "conserved_kmers", "k_high", "k_low", "threshold"]
            for key in keys_to_remove:
                mut_params.pop(key, None)

        elif epoch["mutator"]["type"] == "hotcold":
            current_bias = float(mut_params.get("transition_bias", 2.0))
            mut_params["transition_bias"] = m2.number_input(
                "Transition Bias",
                value=current_bias,
                key=f"mut_bias_{i}"
            )

            st.caption("Hot/Cold Spots Configuration")
            hc1, hc2 = st.columns(2)


            def list_to_str(lst):
                return ", ".join(lst) if isinstance(lst, list) else str(lst)


            def str_to_list(s):
                return [x.strip() for x in s.split(",") if x.strip()]


            current_vars = mut_params.get("variable_kmers", ["AAA", "TTT"])
            current_k_high = mut_params.get("k_high", 10.0)

            new_vars = hc1.text_area("Variable K-mers (Hot)", value=list_to_str(current_vars), help="Comma separated",
                                     key=f"mut_vars_{i}")
            mut_params["variable_kmers"] = str_to_list(new_vars)
            mut_params["k_high"] = hc1.number_input("Hot Factor (k_high)", value=float(current_k_high),
                                                    key=f"mut_kh_{i}")

            current_cons = mut_params.get("conserved_kmers", ["CCC", "GGG"])
            current_k_low = mut_params.get("k_low", 0.1)

            new_cons = hc2.text_area("Conserved K-mers (Cold)", value=list_to_str(current_cons), key=f"mut_cons_{i}")
            mut_params["conserved_kmers"] = str_to_list(new_cons)
            mut_params["k_low"] = hc2.number_input("Cold Factor (k_low)", value=float(current_k_low), key=f"mut_kl_{i}")

            mut_params["threshold"] = st.slider("Similarity Threshold", 0.5, 1.0,
                                                float(mut_params.get("threshold", 0.8)), key=f"mut_thresh_{i}")

        epoch["mutator"]["params"] = mut_params

        st.divider()

        # --- FITNESS LANDSCAPE ---
        st.markdown("#### 🗻 Fitness Landscape")

        f1, f2 = st.columns(2)

        # REMOVED "epistatic" FROM THIS LIST:
        fitness_options = ["purifying", "exposure", "frequency", "neutral", "categorical", "site_specific"]

        current_fit_type = epoch["fitness"].get("type", "purifying")
        if current_fit_type not in fitness_options:
            # If loaded config has 'epistatic', we add it temporarily just to show it, or reset it.
            # Here we append it so it doesn't crash if loading an old config.
            fitness_options.append(current_fit_type)

        new_fit_type = f1.selectbox("Fitness Type", fitness_options, index=fitness_options.index(current_fit_type),
                                    key=f"fit_type_{i}")
        epoch["fitness"]["type"] = new_fit_type

        if "params" not in epoch["fitness"]:
            epoch["fitness"]["params"] = {}
        fit_params = epoch["fitness"]["params"]

        # --- Dynamic Fields & CLEANUP Logic ---

        if new_fit_type == "purifying":
            for k in ["pressure", "update_interval", "site_weights", "site_intensities", "interaction_matrix"]:
                fit_params.pop(k, None)
            fit_params["intensity"] = f2.number_input("Intensity (s)", value=float(fit_params.get("intensity", 0.05)),
                                                      key=f"fit_p_{i}")

        elif new_fit_type == "exposure":
            for k in ["pressure", "site_weights", "site_intensities", "interaction_matrix"]:
                fit_params.pop(k, None)
            fit_params["intensity"] = f2.number_input("Intensity", value=float(fit_params.get("intensity", 0.15)),
                                                      key=f"fit_p_int_{i}")
            fit_params["update_interval"] = st.number_input("Update Interval",
                                                            value=int(fit_params.get("update_interval", 100)),
                                                            key=f"fit_p_upd_{i}")

        elif new_fit_type == "frequency":
            for k in ["intensity", "update_interval", "site_weights", "site_intensities", "interaction_matrix"]:
                fit_params.pop(k, None)
            fit_params["pressure"] = f2.number_input("Immune Pressure", value=float(fit_params.get("pressure", 1.5)),
                                                     key=f"fit_p_pres_{i}")

        elif new_fit_type == "categorical":
            for k in ["intensity", "pressure", "update_interval", "site_intensities", "interaction_matrix"]:
                fit_params.pop(k, None)

            default_weights = "1, 0, 0, 1, 0"
            current_w = fit_params.get("site_weights", default_weights)
            if isinstance(current_w, list) or isinstance(current_w, np.ndarray): current_w = ", ".join(
                map(str, current_w))

            val = st.text_area("Site Weights (Comma separated)", value=str(current_w), key=f"fit_cat_{i}")
            try:
                parsed = [float(x.strip()) for x in val.split(",") if x.strip()]
                fit_params["site_weights"] = parsed
                gl = config["genome"]["length"]
                if len(parsed) != gl:
                    st.error(f"⚠️ Length Mismatch! Genome is {gl}, but you provided {len(parsed)} weights.")
            except:
                st.error("Invalid format for weights.")

        elif new_fit_type == "site_specific":
            for k in ["intensity", "pressure", "update_interval", "site_weights", "interaction_matrix"]:
                fit_params.pop(k, None)

            default_ints = "0.1, 0.5, 0.01"
            current_i = fit_params.get("site_intensities", default_ints)
            if isinstance(current_i, list) or isinstance(current_i, np.ndarray): current_i = ", ".join(
                map(str, current_i))

            val = st.text_area("Site Intensities (Comma separated)", value=str(current_i), key=f"fit_spec_{i}")
            try:
                parsed = [float(x.strip()) for x in val.split(",") if x.strip()]
                fit_params["site_intensities"] = parsed
                gl = config["genome"]["length"]
                if len(parsed) != gl:
                    st.error(
                        f"⚠️ Error: You entered {len(parsed)} values, but Genome Length is {gl}. Please add/remove values.")
            except:
                st.error("Invalid format.")

        # REMOVED THE EPISTATIC ELIF BLOCK HERE

        epoch["fitness"]["params"] = fit_params

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

    # Added "initial_allele" to the list
    sample_types = [
        "diversity",
        "tree",
        "pairwise",
        "haplotype",
        "fitness",
        "fasta",
        "identity",
        "initial_alleles"
    ]

    current_samp_type = sample.get("type", "diversity")
    if current_samp_type not in sample_types:
        current_samp_type = "diversity"

    sample["type"] = cols[0].selectbox(
        "Type",
        sample_types,
        index=sample_types.index(current_samp_type),
        key=f"samp_type_{j}",
        label_visibility="collapsed"
    )

    sample["interval"] = cols[1].number_input(
        "Interval",
        min_value=1,
        value=int(sample.get("interval", 10)),
        key=f"samp_int_{j}",
        label_visibility="collapsed"
    )

    # Auto-generate filename
    sample["file"] = f"output/{sample['type']}.csv"

    # Optional Parameters Handling
    if sample["type"] == "haplotype":
        current_params = sample.get("params", {})
        top_n = cols[2].number_input(
            "Top N",
            min_value=1,
            value=int(current_params.get("top_n", 10)),
            key=f"samp_p_{j}"
        )
        sample["params"] = {"top_n": top_n}

    elif sample["type"] == "initial_alleles":
        # No extra params needed for this sampler based on your code
        cols[2].write("")
        sample["params"] = {}

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

# Tabs for better organization
tab_plots, tab_trees, tab_data = st.tabs([
    "Population Dynamics",
    "Phylogeny & Evolution",
    "Raw Data"
])

# --- TAB 1: DYNAMICS (Plots) ---
with tab_plots:
    st.caption("Visualizing population statistics over time.")
    has_plots = False

    # We iterate over all configured outputs
    for sample in config["sampling"]:
        sample_type = sample.get("type")
        csv_path = sample.get("file")

        # Skip types that belong in other tabs
        if not csv_path or sample_type == "tree" or sample_type == "fasta":
            continue

            # Check for the PNG image generated by finalize()
        png_path = csv_path.replace(".csv", ".png")

        if os.path.exists(png_path):
            has_plots = True

            # Formating the title nicely (e.g. "initial_allele" -> "Initial Allele")
            display_title = sample_type.replace('_', ' ').title()

            # Specific titles for known types (Optional polish)
            if sample_type == "initial_alleles":
                display_title = "Initial Allele Frequencies"
            elif sample_type == "diversity":
                display_title = "Population Diversity"

            st.subheader(display_title)
            st.image(png_path, use_container_width=True)
            st.divider()

    if not has_plots:
        st.info("No dynamics plots found. Run the simulation to generate results.")

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
                    # Show preview of FASTA
                    st.code(f.read(2000) + "\n... (truncated)", language="text")
            else:
                # Show CSV as interactive table
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)

