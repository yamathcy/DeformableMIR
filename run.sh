python main.py sconv=False deform=True retrain=True experiment_name='dcnv2 crt'
python main.py sconv=False deform=True retrain=True mod=False experiment_name='dcnv1 crt'
python main.py sconv=False deform=True retrain=False experiment_name='dcnv2'
python main.py sconv=False deform=True retrain=False mod=False experiment_name='dcnv1'
python main.py sconv=False deform=False retrain=True experiment_name='CNN crt'
python main.py sconv=False deform=False retrain=False experiment_name='CNN'
python main.py sconv=False deform=True retrain=True mel=False experiment_name='dcnv2 crt stft'
python main.py sconv=False deform=True retrain=True mel=False mod=False experiment_name='dcnv1 crt stft'
python main.py sconv=False deform=True retrain=False mel=False experiment_name='dcnv2 stft'
python main.py sconv=False deform=True retrain=False mel=False mod=False experiment_name='dcnv1 stft'
python main.py sconv=False deform=False retrain=False mel=False kernel_size="[(4,1),(16,1),(1,6),(1,16)]" pooling="[2,2,2,2]" experiment_name='OblongCNN'
python main.py sconv=False deform=True retrain=True mel=False kernel_size='[(4,1),(16,1),(1,6),(1,16)]' pooling='[2,2,2,2]' experiment_name='OblongDeformableCNN'