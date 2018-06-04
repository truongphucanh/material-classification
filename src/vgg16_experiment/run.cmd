REM python ./src/vgg16_experiment/split_data.py GTOS 0
REM python ./src/vgg16_experiment/split_data.py GTOS 1
REM python ./src/vgg16_experiment/split_data.py GTOS 2
REM python ./src/vgg16_experiment/split_data.py GTOS 3
REM python ./src/vgg16_experiment/split_data.py GTOS 4
REM python ./src/vgg16_experiment/split_data.py GTOS 5
python .\src\vgg16_experiment\finetune.py GTOS 1 39
python .\src\vgg16_experiment\finetune.py GTOS 2 39
python .\src\vgg16_experiment\finetune.py GTOS 3 39
python .\src\vgg16_experiment\finetune.py GTOS 4 39
python .\src\vgg16_experiment\finetune.py GTOS 5 39
pause