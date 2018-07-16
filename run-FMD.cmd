@echo off
python ./material_classification/source/extract.py keras_vgg16_fc2 FMD original
python ./material_classification/source/extract.py keras_vgg16_fc2 FMD edges
python ./material_classification/source/extract.py keras_vgg16_fc2 FMD texture

python ./material_classification/source/combine_features.py FMD original edges
python ./material_classification/source/combine_features.py FMD original edges texture

python ./material_classification/source/train.py FMD original 1 1
python ./material_classification/source/train.py FMD original 2 2
python ./material_classification/source/train.py FMD original 3 3
python ./material_classification/source/train.py FMD original 4 4
python ./material_classification/source/train.py FMD original 5 5
python ./material_classification/source/train.py FMD original+edges 1 1
python ./material_classification/source/train.py FMD original+edges 2 2
python ./material_classification/source/train.py FMD original+edges 3 3
python ./material_classification/source/train.py FMD original+edges 4 4
python ./material_classification/source/train.py FMD original+edges 5 5
python ./material_classification/source/train.py FMD original+edges+texture 1 1
python ./material_classification/source/train.py FMD original+edges+texture 2 2
python ./material_classification/source/train.py FMD original+edges+texture 3 3
python ./material_classification/source/train.py FMD original+edges+texture 4 4
python ./material_classification/source/train.py FMD original+edges+texture 5 5

python ./material_classification/source/test.py FMD original 1 1
python ./material_classification/source/test.py FMD original 2 2
python ./material_classification/source/test.py FMD original 3 3
python ./material_classification/source/test.py FMD original 4 4
python ./material_classification/source/test.py FMD original 5 5
python ./material_classification/source/test.py FMD original+edges 1 1
python ./material_classification/source/test.py FMD original+edges 2 2
python ./material_classification/source/test.py FMD original+edges 3 3
python ./material_classification/source/test.py FMD original+edges 4 4
python ./material_classification/source/test.py FMD original+edges 5 5
python ./material_classification/source/test.py FMD original+edges+texture 1 1
python ./material_classification/source/test.py FMD original+edges+texture 2 2
python ./material_classification/source/test.py FMD original+edges+texture 3 3
python ./material_classification/source/test.py FMD original+edges+texture 4 4
python ./material_classification/source/test.py FMD original+edges+texture 5 5

python ./material_classification/source/combine_prediction.py FMD test_split_1 original default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py FMD test_split_2 original default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py FMD test_split_3 original default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py FMD test_split_4 original default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py FMD test_split_5 original default_default_default_default original+edges+texture default_default_default_default

python ./material_classification/source/combine_prediction.py FMD test_split_1 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py FMD test_split_2 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py FMD test_split_3 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py FMD test_split_4 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py FMD test_split_5 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default

pause