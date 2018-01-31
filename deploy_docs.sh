echo "Do not forget to build the package before calling this !"
sudo sphinx-build -E -b html docs dist/docs/www
cp docs/app.yaml dist/docs/
cd dist/docs/
gcloud app deploy --project dodoml-docs --quiet --stop-previous-version --version snapshot

echo "https://dodoml-docs.appspot.com/"
echo "http://dodoml.lajavaness.com"
