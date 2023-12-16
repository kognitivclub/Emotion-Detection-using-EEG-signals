import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
from django.conf import settings
def emotion(valence,arousal,dominance,liking,familiarity,relevance):
    file_path= os.path.join(settings.BASE_DIR, 'EEGPROJECT/EEG_DATASET_Chart.csv')
    # data = pd.read_csv('C:\\Users\\revanth\\OneDrive\\Desktop\\EEG Research\\django\\eeg\\EEGPROJECT\\EEG_DATASET_Chart.csv')
    data=pd.read_csv(file_path)
    features = data.drop('emotionCategory', axis=1)
    labels = data['emotionCategory']

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, encoded_labels, test_size=0.05, random_state=15000)
    svm_classifier = SVC(kernel='linear',C=2)

    svm_classifier.fit(X_train, y_train)

    predictions = svm_classifier.predict(X_test)

    decoded_predictions = label_encoder.inverse_transform(predictions)
    
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    input_value=[]
    input_value.append(float(valence))
    input_value.append(float(arousal))
    input_value.append(float(dominance))
    input_value.append(float(liking))
    input_value.append(float(familiarity))
    input_value.append(float(relevance))
    scaled_input = scaler.transform([input_value])
    predicted_label = svm_classifier.predict(scaled_input)
    decoded_label = label_encoder.inverse_transform(predicted_label)
    print("Predicted Emotion Category: {}".format(decoded_label[0]))
    return decoded_label



   
