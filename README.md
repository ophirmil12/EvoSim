# SERVINE: The Evolution Simulator

**SERVINE** (Simplified Evolutionary Run-time Virtual Integrated Nucleotide Environment)
is a high-performance simulation framework designed to model molecular evolution.

By utilizing NumPy-accelerated population dynamics, SERVINE allows researchers to observe
digital genomes as they mutate, compete, and adapt under various selective pressures and
environmental constraints.

---

## 🚀 Key Features

* **Vectorized Population Management:** Handles large populations efficiently using 2D NumPy arrays for genetic storage.
* **Modular Fitness Models:** Supports a wide range of biological scenarios:
* **Frequency Dependent:** Simulates immune escape (Red Queen dynamics).
* **Epistatic:** Models complex gene-gene interactions.
* **Exposure:** Simulates shifting environmental optima.
* **Purifying/Categorical:** Models lethal vs. neutral mutation sites.


* **Advanced Mutation Engine:** Implements nucleotide point mutations with configurable transition/transversion biases.
* **Phylogenetic Reconstruction:** Tracks full ancestry to generate `.nwk` (Newick) tree files and lineage visualizations.
* **Real-time Analytics:** Integrated Streamlit dashboard for visual experimentation.
* **Extensive Sampling:** Automated export of FASTA files, fitness trends, diversity metrics, and haplotype frequencies.

---

## 🛠 Installation

1. **Clone the Repository:**
```bash
git clone https://github.com/ophirmil12/evosim.git
cd evosim
```


2. **Install Dependencies:**
```bash
pip install numpy pandas matplotlib scipy PyYAML streamlit
```



---

## 💻 Usage

### Command Line Interface (CLI)

Run a full simulation based on a YAML configuration file:

**Windows:**

```powershell
$env:PYTHONPATH = "src"
python -m servine.cli src/config_file.yaml
```

**Linux/Mac:**

```bash
export PYTHONPATH=src
python3 -m servine.cli src/config_file.yaml
```

### Graphical User Interface (UI)

Launch the interactive Streamlit dashboard to tune parameters and see results instantly:

```bash
python -m streamlit run UI.py
```

---

## ⚙️ Configuration

The simulator is driven by `config.yaml` files. Below is a snippet of the primary parameters:

```yaml
genome:
  length: 5

population:
  initial_size: 50
  initial_sequence: "ATGCG"

epochs:
  - name: "Baseline_Evolution"
    generations: 500
    mutator:
      type: "nucleotide"
      params:
        rate: 0.003
    fitness:
      type: "purifying"
      params:
        intensity: 0.0005

sampling:
  - type: "diversity"
    interval: 10
    file: "output/example_out/diversity.csv"
  - type: "fasta"
    interval: 100
    file: "output/example_out/final_population.fasta"
  - type: "identity"
    interval: 10
    file: "output/example_out/identity.csv"
  - type: "tree"
    interval: 1
    file: "output/example_out/ancestry.csv"
```

---

## 📂 Project (main) Structure

```text
├── src/servine/
│   ├── evolution/        # Fitness models and mutation logic
│   ├── genome/           # Genome structure and sequence handling
│   ├── io/               # Data samplers (FASTA, CSV, Newick)
│   ├── population/       # NumPy-based population containers
│   ├── simulator.py      # Main simulation engine
│   └── cli.py            # Entry point for configuration-based runs
├── output/               # Generated data and plots
├── UI.py                 # Streamlit dashboard
└── requirements.txt      # Python dependencies
```

---

## 📊 Outputs & Analysis

SERVINE generates a comprehensive suite of data in the specified `output_dir`:

* **`ancestry.nwk`**: Phylogenetic tree in Newick format.
* **`diversity.png`**: Plot of unique genotypes over time.
* **`divergence.csv`**: Raw data showing distance from the founder sequence.
* **`haplotypes.png`**: Visualization of dominant strain frequencies.
* **`sequences.fasta`**: Final population sequences for use in external bioinformatics tools (like MEGA or RAxML).
* And many more!

---

## 📝 License

**Author:** [Ophir Miller](https://www.google.com/search?q=https://github.com/ophirmil12), 
[Batel Eliad](https://github.com/batel418), [Lotem Senderov](https://github.com/lotemsenderov),
[Maya Livshits](https://github.com/mayalivshits)

**Project Link:** [https://github.com/ophirmil12/evosim](https://www.google.com/search?q=https://github.com/ophirmil12/evosim)