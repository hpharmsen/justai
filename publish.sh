start_time=$(date +%s)
uv pip install twine build
/bin/rm -f dist/*
export VERSION=`python bumpversion.py -v patch`
python -m build
twine upload dist/*
git commit -v -a -m "publish `date`"
git tag -a $VERSION -m "version $VERSION"
git push origin main
git push origin $VERSION
duration=$(($(date +%s) - start_time))
echo "${GREEN}Published in $duration secs${NC}"
echo ""
echo "run:"
echo "uv pip install justai==$VERSION"
