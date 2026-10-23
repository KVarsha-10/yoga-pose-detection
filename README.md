# Namaste Yoga Tutor

Flask web app for webcam-based yoga pose guidance. The model and webcam
detection run locally; voice prompts use the browser's speech engine.

## Run locally

From this project folder in PowerShell:

```powershell
.\venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000> in a browser, allow camera access if prompted,
choose a pose sequence, and select **Start my practice**. Starting with a click
also enables the browser's spoken prompts.

## Project layout

```text
app.py                 Flask routes, webcam session, pose state and API
templates/index.html   Page structure
static/app.css         Responsive app styling
static/app.js          Sequence builder, browser speech, and live UI updates
pose_data.csv          Labeled pose landmark examples
model/                 Model training code and saved model (local file)
pose_detection/        MediaPipe pose detector
utils/                 Landmark features, feedback, and reference importer
pose_references/       Optional pose reference images
requirements.txt       Python dependencies
venv/                  Local virtual environment; do not copy to Git
```

The saved model file is stored locally at `model/yoga_model.pkl` and is excluded
from Git because of its size. Reference images use lowercase pose names with
underscores, such as `adho_mukha_svanasana.jpg`.
