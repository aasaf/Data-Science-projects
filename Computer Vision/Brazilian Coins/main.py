from functions import build_model,extract_labels,convert_image_files_to_array
from functions import detect_many_coins_for_regression,detect_classification_coins
#from Include import functions
from pathlib import Path

#all classification inputs
input_path = Path(r'C:\Users\Asaf\Desktop\brazilian coins\classification_dataset\all')

#all classification image files
image_files = list(input_path.glob('*.jpg'))


#use this funcion just once to detect the coins in classification images for trainning the model
'''detect_classification_coins(image_files)'''

#all regression inputs
input_regression_path = Path(r'C:\Users\Asaf\Desktop\brazilian coins\regression_dataset\all')

#all regression image files
image_regression_files = list(input_path.glob('*.jpg'))

#build the model

classifier, callbacks = build_model()

#train the model

input_path_all_labels = Path(r'C:\Users\Asaf\Desktop\brazilian coins\valid_classification_dataset\training set\all_data_labels2')
y_labels,x = extract_labels(input_path_all_labels)
X_train, X_test, y_train, y_test = train_test_split(x,y_labels, test_size=0.33, random_state=42)
x_train_list = convert_image_files_to_array(X_train)
classifier.fit(x=x_train_list, y=y_train, validation_split=0.3,batch_size=32,
               epochs = 30,  callbacks = callbacks)

input_regression_path = Path(r'C:\Users\Asaf\Desktop\brazilian coins\regression_dataset\all')
#all classification image files
image_regression_files = list(input_regression_path.glob('*.jpg'))

y_pred,y_true  = detect_many_coins_for_regression(image_regression_files, classifier)
r2_score(y_true, y_pred)

final_result = np.array(y_pred) - np.array(y_true)
accuracy = sum(final_result==0)/len(final_result)
print("accuracy: ", accuracy)
print(classification_report(y_true,y_pred))

