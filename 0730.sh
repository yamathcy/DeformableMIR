python main.py sconv=False deform=True retrain=True experiment_name='dcnv2 crt'
python main.py sconv=False deform=True retrain=True mod=False experiment_name='dcnv1 crt'
python main.py sconv=False deform=True retrain=False experiment_name='dcnv2'
python main.py sconv=False deform=True retrain=False mod=False experiment_name='dcnv1'
python main.py sconv=True deform=True retrain=True kernel_size=[(5,1),(15,1),(1,5),(1,15)] pooling=[2,2,2,2] experiment_name='dcnv3 crt oblong'
python main.py sconv=True deform=True retrain=True kernel_size=[3,5,3,5] pooling=[2,2,2,2] experiment_name='dcnv3 crt same_amount'
python main.py sconv=False deform=True retrain=True experiment_name='dcnv2 crt pool' pooling=[2,2,2,2]
python main.py sconv=False deform=True retrain=False experiment_name='dcnv2 pool' pooling=[2,2,2,2]