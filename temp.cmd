@echo off
python ./src/train.py FMD original 1 1
python ./src/train.py FMD original 2 2
python ./src/train.py FMD original 3 3
python ./src/train.py FMD original 4 4
python ./src/train.py FMD original 5 5

python ./src/train.py FMD original+edges+texture 1 1
python ./src/train.py FMD original+edges+texture 2 2
python ./src/train.py FMD original+edges+texture 3 3
python ./src/train.py FMD original+edges+texture 4 4
python ./src/train.py FMD original+edges+texture 5 5

python ./src/train.py FMD original+edges 1 1
python ./src/train.py FMD original+edges 2 2
python ./src/train.py FMD original+edges 3 3
python ./src/train.py FMD original+edges 4 4
python ./src/train.py FMD original+edges 5 5

pause