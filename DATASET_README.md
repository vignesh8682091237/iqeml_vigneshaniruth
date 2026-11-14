# Dataset: Open University Learning Analytics Dataset (OULAD)

This package does NOT contain OULAD raw data or any per-student data. OULAD is licensed by its owner (The Open University) and must be obtained directly.

## Obtain
Download from https://analyse.kmi.open.ac.uk/open-dataset and accept its licence. Put the seven CSV files in one folder: assessments.csv, courses.csv, studentAssessment.csv, studentInfo.csv, studentRegistration.csv, studentVle.csv, vle.csv.

## Prepare
    cd experimental_code
    export IQEL_RAW=/path/to/that/folder
    python prep_compact.py     # restricts clicks to day <= 84, writes per-student tables
    python make_week_data.py   # builds feature arrays for Weeks 4, 8, 12

Target T2 = Withdrawn+Fail vs Pass+Distinction. Students withdrawn by a checkpoint are excluded. Test presentation = 2014J.

## Included (aggregate only)
results/ holds model outputs (probabilities per held-out row) and metrics. No identifiers.
