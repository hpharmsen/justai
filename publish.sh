# Aborteer bij de eerste fout. Zonder dit liep het script door na een mislukte
# build en upload, en pushte het alsnog een tag voor een versie die nooit
# gebouwd is. Geen set -u: ${GREEN}/${NC} hieronder zijn niet gedefinieerd.
set -e
start_time=$(date +%s)
uv pip install twine build
/bin/rm -f dist/*
export VERSION=`uv run python bumpversion.py -v minor`
# uv run, niet kaal python/twine: die landen via PATH op de systeem-Python en
# niet in de venv waar de regel hierboven build en twine net heeft gezet.
uv run python -m build

# Controleer dat de wheel elk .py-bestand uit justai/ bevat. Een .gitignore-regel
# kan stilzwijgend bronbestanden uit de build laten vallen (hatchling volgt
# .gitignore), en dat merk je anders pas als een gebruiker het pakket installeert.
uv run python - <<'EOF'
import glob, sys, zipfile
from pathlib import Path

wheel = max(glob.glob('dist/*.whl'))
in_wheel = {n for n in zipfile.ZipFile(wheel).namelist() if n.endswith('.py')}
on_disk = {str(p) for p in Path('justai').rglob('*.py') if '__pycache__' not in str(p)}
missing = sorted(on_disk - in_wheel)
if missing:
    print(f'ABORT: {len(missing)} bronbestanden ontbreken in {wheel}:', file=sys.stderr)
    for m in missing[:10]:
        print(f'  {m}', file=sys.stderr)
    print('Kijk in .gitignore naar een patroon dat te breed matcht.', file=sys.stderr)
    sys.exit(1)
print(f'{wheel}: alle {len(on_disk)} bronbestanden aanwezig')
EOF

uv run twine upload dist/*
git commit -v -a -m "publish `date`"
git tag -a $VERSION -m "version $VERSION"
git push origin main
git push origin $VERSION
duration=$(($(date +%s) - start_time))
echo "${GREEN}Published in $duration secs${NC}"
echo ""
echo "run:"
echo "uv pip install justai==$VERSION"
