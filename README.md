# rsna_lstm_retrain

##Requirements
###Python 3

##Installation
###pip install -r requirements.txt

##Sample API usage

###curl 127.0.0.1:8484/trainingCSVHeaders
###{"headers":"examDescription,OtherHx,OtherSx,OtherSx2,indicationDescription,label_binary"}

###curl -F 'data=@PORT_CHEST-annotations-combined_0_to_1500.csv' http://127.0.0.1:8484/retrain
###{"message":"retraining complete. model_version: ml_postproc_lstm_2019-03-29_04:16:18_data"}
  
###curl -d '{"examDescription":"PORT CHEST 1 VIEW", "OtherHX":"Hyperkalemia [E87.5]", "OtherSX":"WEAKNESS", "OtherSX2":"s/p RIJ","indicationDescription":"Acute kidney failure, unspecified"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:8484/predict
###{"model_version":"ml_postproc_lstm_2019-03-29_04:18:19_data","prediction":"0.75536704"}

###curl -d '{"model_version":"2019-03-29_03:59:41_data"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:8484/load_model

###deletes latest model and loads previous model
###curl -X POST  http://127.0.0.1:8484/undo_retrain
###{"message":"undo complete. model_version: ml_postproc_lstm_2019-03-29_03:59:41_data"}

###curl http://127.0.0.1:8484/model_version
###{"message":"ml_postproc_lstm_2019-03-29_03:59:41_data"}
