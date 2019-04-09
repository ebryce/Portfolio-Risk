
import matplotlib.pyplot as plt, numpy as np
import itertools
from sklearn.metrics import explained_variance_score, mean_squared_error, confusion_matrix, classification_report, accuracy_score


def clf_model_eval(y_true, y_pred, classes=[0,1], cmap=plt.cm.Blues):
    '''
    Print classification repot and plot normalized confusion matrix.
    '''
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(classification_report(y_true, y_pred))
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = plt.figure(figsize=(5,5))
    fig.patch.set_facecolor('white')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()