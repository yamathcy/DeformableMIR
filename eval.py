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
import numpy as np
from tqdm import tqdm


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
def deep_feature_dump(loader, model, dump_name=None):
    labels = []
    features = []
    print("dumping...")
    model.to(DEVICE)
    for sig, la, _ in tqdm(loader):
        sig = sig.to(DEVICE)
        la = int(la)
        labels.append(la)
        feature = model.get_feature(sig)
        feature = feature.detach().cpu().numpy().copy()
        feature = np.squeeze(feature)
        features.append(feature)
    
    features = np.array(features)
    labels = np.array(labels)

    deepfeature = (features, labels)
    #with open("{}_deep_feature.joblib".format(dump_name), mode="wb") as f:
        #joblib.dump(deepfeature, f, compress=3)
        
    return deepfeature



def embed_visualize(dataset, plot_title, target_class_inv, mode='umap'):
    # dataset must be set of (extracted_feature, label)
    # visualize tsne deep
    sns.set()
    squeezed = []
    for train_vec in dataset[0]:
        train_vec = np.squeeze(train_vec)
        squeezed.append(train_vec)

    if mode == 'tsne':
        mapper = TSNE(n_components=2, random_state=1)
        emb = mapper.fit_transform(squeezed)

    else:
        mapper = UMAP(random_state=1)
        emb = mapper.fit_transform(squeezed)

    np_label = np.array([dataset[1]])
    #print(np_label.T.shape)

    dat = np.hstack((emb, np_label.T))
    df = pd.DataFrame(dat,columns=["x","y","label"])
    df = df.astype({'label': int})
    df["label"] = df["label"].replace(target_class_inv)
    #print(df.head())
    plt.figure(figsize=(7,5))
    sp=sns.scatterplot(data=df,palette="bright",x="x",y="y",hue="label", alpha=.5, linewidth=0.1)
    plt.title("Embedding" + str(plot_title))
    # plt.scatter(x_tsne_deep[:,0], x_tsne_deep[:,1], c=deep_feature_test[1],alpha=0.8, cmap="rainbow")
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    plt.tight_layout()
    fig=sp.get_figure()
    fig.savefig("tsne_{}.png".format(plot_title))
    mlflow.log_artifact("tsne_{}.png".format(plot_title))


def evaluation_wandb(logger:WandbLogger, test_loader, model, plot_title, random_state, target_class, target_class_inv, retrain_loader=None,retrain=False, epoch=10, SSL=False):
    feature_vecs = []
    pred = []
    pre_prob = []
    label = []
    if retrain:
        # retrain 
        print(model.children())
        for params in model.parameters():
            params.requires_grad = False
        model.net.classifier.weight.requires_grad = True
        model.net.classifier.bias.requires_grad = True
        model.retrain=True
        # model.train()
        fe_trainer = pl.Trainer(max_epochs=epoch,precision=32)
        model.lr=1e-5
        fe_trainer.fit(model, retrain_loader)
    model.eval()
    print("embed")

    for sig, la in tqdm(test_loader):
        sig = sig.to("cuda")

        la = int(la)
        model = model.to("cuda")
        _ , feature = model(sig)
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

    # else:
    #     dump_train_data = deep_feature_dump(train_loader, model)
    #     dump_test_data = deep_feature_dump(test_loader, model)
    #     print('non-deep backend')
    #     if classfier_method == 'randomforest':
    #         clf = BalancedRandomForestClassifier(class_weight='balanced',random_state=2023, n_jobs=-1, n_estimators=100)

    #     elif classfier_method == 'svm':
    #         clf = SVC(class_weight='balanced', probability=True)
    #     clf.fit(dump_train_data[0], dump_train_data[1])
        
    #     # fit on train_data[0] : data, train_data[1]: label
    #     # predict on test data
    #     pred = clf.predict(dump_test_data[0])
    #     pre_prob = clf.predict_proba(dump_test_data[0])
    #     label = dump_test_data[1]
    #     feature_vecs = dump_test_data[0]
    try:
        embed_visualize((feature_vecs, label), plot_title, target_class_inv)
    except:
        pass
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
    
    return macrof1, accuracy, balanced, top_2, top_3, df_cmx, report_dict
