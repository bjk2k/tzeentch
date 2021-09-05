from tensorflow.keras.callbacks import Callback
from twilio.rest import Client

account_sid = ""
auth_token = ""

client = Client(account_sid, auth_token)


class NotificationCallback(Callback):
    def __init__(self, model_id):
        self.model_identification = model_id
        super().__init__()

    def on_train_end(self, logs=None):
        msg = client.messages.create(
                body=f"training completed for {self.model_identification} with {logs['loss']}",
                from_="+4915228760197",
                to="+4915228760197"
        )
