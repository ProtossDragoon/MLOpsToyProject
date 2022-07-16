import bentoml
import logging
import re
from bentoml.io import Text, JSON

bentoml_logger = logging.getLogger("bentoml")

class spamdetectionrunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ()
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        # load the model back:
        classifier = bentoml.picklable_model.load_model("spam_detector:latest")
        self.p_spam, self.p_ham, self.parameters_spam, self.parameters_ham = classifier
        
    @bentoml.Runnable.method(batchable=False)
    def is_spam (self, input_text):

        message = re.sub('\W', ' ', input_text)
        message = message.lower().split()

        p_spam_given_message = self.p_spam
        p_ham_given_message = self.p_ham
       
        for word in message:
            if word in self.parameters_spam:
                p_spam_given_message *= self.parameters_spam[word]

            if word in self.parameters_ham:
                p_ham_given_message *= self.parameters_ham[word]

        if p_ham_given_message < p_spam_given_message:
            return [True, p_spam_given_message, p_ham_given_message]
        else:
            return [False, p_spam_given_message, p_ham_given_message]
        

spam_detection_runner = bentoml.Runner(spamdetectionrunnable)

svc = bentoml.Service('spam_detector', runners=[spam_detection_runner])

@svc.api(input=Text(), output=JSON())
def analysis(input_text):
    is_spam, p_spam_given_message,  p_ham_given_message= spam_detection_runner.is_spam.run(input_text)
    return { "is_spam": is_spam, "P(Spam|message)": p_spam_given_message, "P(Ham|message)": p_ham_given_message}
