@echo off
python ./material_classification/source/extract.py keras_vgg16_fc2 GTOS original
python ./material_classification/source/extract.py keras_vgg16_fc2 GTOS edges
python ./material_classification/source/extract.py keras_vgg16_fc2 GTOS texture

python ./material_classification/source/combine_features.py GTOS original edges
python ./material_classification/source/combine_features.py GTOS original edges texture

python ./material_classification/source/train.py GTOS original 1 1
python ./material_classification/source/train.py GTOS original 2 2
python ./material_classification/source/train.py GTOS original 3 3
python ./material_classification/source/train.py GTOS original 4 4
python ./material_classification/source/train.py GTOS original 5 5
python ./material_classification/source/train.py GTOS original+edges 1 1
python ./material_classification/source/train.py GTOS original+edges 2 2
python ./material_classification/source/train.py GTOS original+edges 3 3
python ./material_classification/source/train.py GTOS original+edges 4 4
python ./material_classification/source/train.py GTOS original+edges 5 5
python ./material_classification/source/train.py GTOS original+edges+texture 1 1
python ./material_classification/source/train.py GTOS original+edges+texture 2 2
python ./material_classification/source/train.py GTOS original+edges+texture 3 3
python ./material_classification/source/train.py GTOS original+edges+texture 4 4
python ./material_classification/source/train.py GTOS original+edges+texture 5 5

python ./material_classification/source/test.py GTOS original 1 1
python ./material_classification/source/test.py GTOS original 2 2
python ./material_classification/source/test.py GTOS original 3 3
python ./material_classification/source/test.py GTOS original 4 4
python ./material_classification/source/test.py GTOS original 5 5
python ./material_classification/source/test.py GTOS original+edges 1 1
python ./material_classification/source/test.py GTOS original+edges 2 2
python ./material_classification/source/test.py GTOS original+edges 3 3
python ./material_classification/source/test.py GTOS original+edges 4 4
python ./material_classification/source/test.py GTOS original+edges 5 5
python ./material_classification/source/test.py GTOS original+edges+texture 1 1
python ./material_classification/source/test.py GTOS original+edges+texture 2 2
python ./material_classification/source/test.py GTOS original+edges+texture 3 3
python ./material_classification/source/test.py GTOS original+edges+texture 4 4
python ./material_classification/source/test.py GTOS original+edges+texture 5 5

python ./material_classification/source/combine_prediction.py GTOS test_split_1 original default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py GTOS test_split_2 original default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py GTOS test_split_3 original default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py GTOS test_split_4 original default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py GTOS test_split_5 original default_default_default_default original+edges+texture default_default_default_default

python ./material_classification/source/combine_prediction.py GTOS test_split_1 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py GTOS test_split_2 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py GTOS test_split_3 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py GTOS test_split_4 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default
python ./material_classification/source/combine_prediction.py GTOS test_split_5 original default_default_default_default original+edges default_default_default_default original+edges+texture default_default_default_default

pause