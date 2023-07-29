import torch
from torch import device
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from sklearn.svm import SVC
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from umap import UMAP
from tqdm import tqdm


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def evaluation_wandb(logger:WandbLogger, model, train_loader, test_loader, plot_title, classfier_method, random_state, class_size, class_weight, retrain_loader=None,retrain=False, epoch=10, SSL=False):
    if classfier_method=='nn': 
        feature_vecs = []
        pred = []
        pre_prob = []
        label = []
        if retrain:
            # retrain 
            print(model.children())
            for param in model.parameters():
                param.requires_grad = False
            model.output.weight.requires_grad = True
            model.output.bias.requires_grad = True
            # model.train()
            fe_trainer = pl.Trainer(max_epochs=epoch,precision=16)
            model.configure_optimizers(lr=1e-5)
            fe_trainer.fit(model, retrain_loader)
        model.eval()
        print("embed")
        for sig, la, _ in tqdm(test_loader):
            with torch.no_grad():
                #print(sig.shape)
                sig = sig.to(DEVICE)
                la = int(la)
                model.to(DEVICE)
                feature = model.get_feature(sig)
                feature = feature.detach().cpu().numpy().copy()
                out = model.predict(sig)
                prob= model.predict_proba(sig)
                pred.append(out)
                pre_prob.append(prob)
                label.append(la)
                feature_vecs.append(feature)

        pred = np.array(pred)
        pre_prob = np.array(pre_prob)
        label = np.array(label)
        # todo: delete
        #pred = model.predict_classes(test_data[0])
        #pre_prob = model.predict_proba(test_data[0])

    else:
        dump_train_data = deep_feature_dump(train_loader, model)
        dump_test_data = deep_feature_dump(test_loader, model)
        print('non-deep backend')
        if classfier_method == 'randomforest':
            clf = BalancedRandomForestClassifier(class_weight='balanced',random_state=SEED, n_jobs=-1, n_estimators=100)

        elif classfier_method == 'svm':
            clf = SVC(class_weight='balanced', probability=True)
        clf.fit(dump_train_data[0], dump_train_data[1])
        
        # fit on train_data[0] : data, train_data[1]: label
        # predict on test data
        pred = clf.predict(dump_test_data[0])
        pre_prob = clf.predict_proba(dump_test_data[0])
        label = dump_test_data[1]
        feature_vecs = dump_test_data[0]
    
    embed_visualize((feature_vecs, label), plot_title)
    accuracy = accuracy_score(y_true=label, y_pred=pred)
    balanced = balanced_accuracy_score(y_true=label, y_pred=pred)
    top_2 = top_k_accuracy_score(k=2,y_score=pre_prob, y_true=label)
    top_3 = top_k_accuracy_score(k=3,y_score=pre_prob, y_true=label)
    macrof1 = f1_score(y_true=label,y_pred=pred, average='macro')
    microf1 = f1_score(y_true=label,y_pred=pred, average='micro')
    weighted_f1 = f1_score(y_true=label,y_pred=pred, average='weighted')
    report = classification_report(y_true=label, y_pred=pred, target_names=list(target_class.keys()))
    report_dict = classification_report(y_true=label, y_pred=pred, target_names=list(target_class.keys()), output_dict=True)
    print(report)
    cf_data_d=confusion_matrix(label, pred)
    print(cf_data_d)
    df_cmx = pd.DataFrame(cf_data_d, index=target_class.keys(), columns=target_class.keys())
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix of {}".format(plot_title))
    print("---Accuracy Report---")
    print("Overall accracy:{:.3f}".format(accuracy))
    print("Overall balanced accracy:{:.3f}".format(balanced))
    print("Top-2:{:.3f}".format(top_2))
    print("Top-3:{:.3f}".format(top_3))
    print("f1-score: {:.3f}".format(macrof1))
    plt.savefig("confusion_matrix.png")
    logger.log_image(key='confusion_matrix', images=['confusion_matrix.png'])

    plt.show()
    with open(plot_title + "_result.txt", 'a') as f:
        print("Random state: {}".format(random_state),file=f)
        print("---Accuracy Report---", file=f)
        print("Overall accracy:{:.3f}".format(accuracy) ,file=f)
        print("Overall balanced accracy:{:.3f}".format(balanced), file=f)
        print("Top-2:{:.3f}".format(top_2), file=f)
        print("Top-3:{:.3f}".format(top_3),file=f)
        print("f1-score: {:.3f}".format(macrof1))
        print(report, file=f)
        try:
            lw = model.get_layer_weight()
            print(lw,file=f)
            print("written layer weights")
            plt.plot(lw)
            plt.tight_layout()
            plt.savefig("layer_weight.png")
            # mlflow.log_artifact('layer_weight.png')
        except:
            print("weight failed")
            pass
    
    return macrof1, accuracy, balanced, top_2, top_3, df_cmx, report_dict
