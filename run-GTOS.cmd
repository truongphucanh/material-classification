@echo off
python ./src/test.py GTOS original 1 1
python ./src/test.py GTOS original+edges+texture 1 1
python ./src/combine_prediction.py GTOS test_split_1 original default_default_default_default original+edges+texture default_default_default_default

rem python ./src/test.py GTOS original 2 2
rem python ./src/test.py GTOS original+edges+texture 2 2
rem python ./src/combine_prediction.py GTOS test_split_2 original default_default_default_default original+edges+texture default_default_default_default

rem python ./src/test.py GTOS original 3 3
rem python ./src/test.py GTOS original+edges+texture 3 3
rem python ./src/combine_prediction.py GTOS test_split_3 original default_default_default_default original+edges+texture default_default_default_default

rem python ./src/test.py GTOS original 4 4
rem python ./src/test.py GTOS original+edges+texture 4 4
rem python ./src/combine_prediction.py GTOS test_split_4 original default_default_default_default original+edges+texture default_default_default_default

rem python ./src/test.py GTOS original 5 5
rem python ./src/test.py GTOS original+edges+texture 5 5
rem python ./src/combine_prediction.py GTOS test_split_5 original default_default_default_default original+edges+texture default_default_default_default

pause