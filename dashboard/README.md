# Interactive Demo Dashboard

This dependency-free dashboard reads the project's existing CSV/JSON outputs
and presents the thesis evidence chain as an interactive web demo.

Refresh the embedded data and start the local server with one command:

```bash
python3 dashboard/run_dashboard.py
```

Open `http://localhost:8765`.

The **Start guided demo** button automatically cycles through the recommended
video-demo sequence.

To refresh only the data bundle:

```bash
python3 dashboard/generate_dashboard_data.py
```
