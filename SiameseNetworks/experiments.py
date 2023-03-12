# trouvé comment faire pour créer un nouveau dataset et y mettre les données dans les bonnes dimensions !! 
maintenant, adapter ça au code existant pour pouvoir appliquer le modèle sur les données préprocessées

from datasets.dataset_dict import DatasetDict
from datasets import Dataset

n_train = dailydialog['train'].num_rows
n_val = dailydialog['validation'].num_rows
n_test = dailydialog['test'].num_rows

# x_train = dailydialog['train']['text']
# x_val = dailydialog['validation']['text']
# x_test = dailydialog['test']['text']
# y_train = dailydialog['train']['label']
# y_val = dailydialog['validation']['label']
# y_test = dailydialog['test']['label']

x_train = np.array(dailydialog['train']['text']).reshape((12*n_train, 20))
x_val = np.array(dailydialog['validation']['text']).reshape((12*n_val, 20))
x_test = np.array(dailydialog['test']['text']).reshape((12*n_test, 20))
y_train = np.array(dailydialog['train']['label']).reshape((-1,1))
y_val = np.array(dailydialog['validation']['label']).reshape((-1,1))
y_test = np.array(dailydialog['test']['label']).reshape((-1,1))

d = {'train':Dataset.from_dict({'label':y_train,'text':x_train}),
     'val':Dataset.from_dict({'label':y_val,'text':x_val}),
     'test':Dataset.from_dict({'label':y_test,'text':x_test})
     }

DatasetDict(d)
