python train.py keras_vgg16_fc2-original 1 1
python test.py keras_vgg16_fc2-original 1 1

python train.py keras_vgg16_fc2-original+keras_vgg16_fc2-edges 1 1
python test.py keras_vgg16_fc2-original+keras_vgg16_fc2-edges 1 1

python combine_prediction.py testlist01 keras_vgg16_fc2-original default_default_default_default keras_vgg16_fc2-original+keras_vgg16_fc2-edges default_default_default_default
python combine_prediction.py testlist01 keras_vgg16_fc2-original linear_0.1_None_None keras_vgg16_fc2-original+keras_vgg16_fc2-edges linear_0.1_None_None
python combine_prediction.py testlist01 keras_vgg16_fc2-original linear_1_None_None keras_vgg16_fc2-original+keras_vgg16_fc2-edges linear_1_None_None
python combine_prediction.py testlist01 keras_vgg16_fc2-original poly_1000_2_None keras_vgg16_fc2-original+keras_vgg16_fc2-edges poly_1000_2_None
python combine_prediction.py testlist01 keras_vgg16_fc2-original poly_1000_3_None keras_vgg16_fc2-original+keras_vgg16_fc2-edges poly_1000_3_None
python combine_prediction.py testlist01 keras_vgg16_fc2-original rbf_1000_None_0.000001 keras_vgg16_fc2-original+keras_vgg16_fc2-edges rbf_1000_None_0.000001
python combine_prediction.py testlist01 keras_vgg16_fc2-original rbf_1000_None_0.0001 keras_vgg16_fc2-original+keras_vgg16_fc2-edges rbf_1000_None_0.0001
