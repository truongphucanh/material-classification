python train.py keras_vgg16_fc2-original 0 0
python test.py keras_vgg16_fc2-original 0 0

python train.py keras_vgg16_fc2-original+keras_vgg16_fc2-edges 0 0
python test.py keras_vgg16_fc2-original+keras_vgg16_fc2-edges 0 0

python combine_prediction.py testlist00 keras_vgg16_fc2-original default_default_default_default keras_vgg16_fc2-original+keras_vgg16_fc2-edges default_default_default_default
pause