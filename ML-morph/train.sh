python preprocessing.py -i ./cropped/finger -t all.tps
python ./ml-morph/shape_trainer.py -d train.xml -t test.xml -th 4 -n 2000 -nu 0.0001 -c 20 -dp 5
python ./ml-morph/prediction.py -i test -p predictor.dat
imglab output.xml

