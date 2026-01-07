# Stop ved første fejl
$ErrorActionPreference = "Stop"

Write-Host "=== ECG Classifier: Clean install & smoke test ==="

# Gå til repo root (scriptet ligger i tests/)
Set-Location $PSScriptRoot\..

# Deaktiver evt. aktivt venv
if ($env:VIRTUAL_ENV) {
    deactivate
}

# Fjern eksisterende test-venv
if (Test-Path ".venv-test") {
    Remove-Item -Recurse -Force ".venv-test"
}

# Opret nyt venv
py -3.11 -m venv .venv-test

# Aktivér test-venv
. .venv-test\Scripts\Activate.ps1

# Opgradér pip
python -m pip install --upgrade pip

# Installer pakken fra repo root
python -m pip install .

# ---------- Basic CLI sanity ----------

Write-Host "`n=== CLI help ==="
ecg-classifier --help

# ---------- Demo inference (bundled data) ----------

Write-Host "`n=== Demo: CSV + Logistic Regression ==="
ecg-classifier demo --format csv --model logreg

Write-Host "`n=== Demo: CSV + GRU ==="
ecg-classifier demo --format csv --model gru

Write-Host "`n=== Demo: WFDB + Logistic Regression ==="
ecg-classifier demo --format wfdb --model logreg

Write-Host "`n=== Demo: WFDB + GRU ==="
ecg-classifier demo --format wfdb --model gru

# ---------- Run inference with explicit input ----------

# Brug CSV demo-filen via CLI-kontrakt
$csv_demo = "src/ecg_classifier/demo/test_ecg_12lead.csv"

Write-Host "`n=== Run: CSV + Logistic Regression ==="
ecg-classifier run --input $csv_demo --format csv --model logreg

Write-Host "`n=== Run: CSV + GRU ==="
ecg-classifier run --input $csv_demo --format csv --model gru

$wfdb_demo = "src/ecg_classifier/demo/wfdb/demo_wfdb"

Write-Host "`n=== Run: WFDB + Logistic Regression ==="
ecg-classifier run --input $csv_demo --format csv --model logreg

Write-Host "`n=== Run: CSV + GRU ==="
ecg-classifier run --input $csv_demo --format csv --model gru



# ---------- Cleanup ----------

deactivate

Write-Host "`n=== Smoke test completed successfully ==="
