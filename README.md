# ( Regression ) project :
 # ( New York City Taxi Trip Duration ) : kaggle competition
 
## First: i made ( BaseLine_Model ) : i wrote All Details on ( BaseLine_Model.ipynb ) jupiter file 
  - ( Date Preparation --> EDA --> Feature Engineering --> Model Training --> Model Validation & Evaluation --> Tuning & finalize )
  - in ( EDA ) : i visualize and analayzed the data and i made desicion on the Features.
  - i used ( RandomForestRegressor ) → algorithm for regression.
  - the accuracy is nearly ( 70 % )
  
## Second: i made ( advancedModel ) , to Enhance the Berformance , so i used:
  - ( XGBoost Regressor ) --> to inhance the model.
  - in ( Feature Engineering ) --> i add new features to make strong crrelated realtion with the Target.
  - i made deep ( Data Cleaning ).
  - the accuracy became nearly ( 98 % ). 

## Third: i made ( Kaggle_TestData_Preparation.py ) : to prepare submission file to send it to kaggle for apply Testing Score  
   - MY Score: ( 0.47672 ) ---> metric for this competition is Root Mean Squared Logarithmic Error ( RMSE ) , i so proud for this.

## Final: i made ( Map_App ) --> this is Web contain Map (Preferably applied on :( NYC ) )
   - in the terminal --> Must write command (streamlit run path/Map_App.py) as similar this command ( streamlit run d:/Code_ML/Project_TaxiTripDuration/Map_App.py ) to give you the Web .
   - select ( Pickup ) and ( Dropoff ) on the Map , then click on ( Predictioin Duration ) --> then it will predict the Time Duration for the Distance .

https://www.linkedin.com/posts/anas-tamer-086821267_%D8%A7%D9%84%D8%B3%D9%84%D8%A7%D9%85-%D8%B9%D9%84%D9%8A%D9%83%D9%85-%D8%AD%D8%A7%D8%A8%D8%A8-%D8%A3%D8%B4%D8%A7%D8%B1%D9%83-%D9%85%D8%B9%D8%A7%D9%83%D9%85-%D9%85%D9%84%D8%AE%D8%B5-%D8%B4%D8%BA%D9%84%D9%8A-%D9%81%D9%8A-activity-7358898621020364802-rCKu?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAEFweSYBAYnONkphZR-zPib5pYV7nTxOurE&utm_campaign=copy_link

 
<img width="1266" height="607" alt="image" src="https://github.com/user-attachments/assets/3a3196c6-ba46-446d-92f3-d6571470b5a7" />
