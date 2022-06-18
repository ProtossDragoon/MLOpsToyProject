
# import the IrisClassifier class defined above
from spam_classifier import SpamClassifier
# import SpamClassifier
import joblib
# Create a iris classifier service instance
spam_classifier_service = SpamClassifier()

file_name = './dbfs/my_project_models/model.pkl'
naive_bayes_loaded = joblib.load(file_name) 
    
# Pack the newly trained model artifact
spam_classifier_service.pack('model', naive_bayes_loaded)

# Save the prediction service to disk for model serving
saved_path = spam_classifier_service.save()