# Pose reference images

Place one reference photo or diagram per pose in this folder. The Flask tutor shows
the current pose's image in the right-hand panel when its file is available.

Use a lowercase filename with spaces replaced by underscores, for example:

- `adho_mukha_svanasana.jpg`
- `virabhadrasana_two.png`

Supported formats are `.jpg`, `.jpeg`, `.png`, and `.webp`. The image should clearly
show one person demonstrating the matching asana. If an image is missing, the
tutor displays a reference-photo placeholder in the panel.

The project currently contains no pose reference images. One dataset with the
same 47 labels is [Roboflow Universe's yoga_pose_classifier dataset by
Codemonk](https://universe.roboflow.com/codemonk/yoga_pose_classifier), listed
under CC BY 4.0. Follow its attribution and license terms when using its images.

To import sample images from an extracted classification dataset, run this from
the project root (replace the folder with the location where you extracted it):

```powershell
.\venv\Scripts\python.exe utils\import_pose_references.py "C:\path\to\extracted-dataset"
```

The importer looks for class-named folders such as `train\Padmasana`, then
copies one image per matching pose into this folder using the filenames above.
The tutor uses those images in its side panel. Check the printed list and
confirm each selected sample clearly demonstrates its named pose.
