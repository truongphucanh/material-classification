@echo off
python ./src/extract.py keras_vgg16_fc2 FMD texture
python ./src/combine_features.py FMD original edges
python ./src/combine_features.py FMD original edges texture
pause