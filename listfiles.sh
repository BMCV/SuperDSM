mkdir .tmpdst
python setup.py install --root .tmpdst --record .listfiles.txt
rm -rf .tmpdst
less listfiles.txt
rm .listfiles.txt
