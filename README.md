  # i will summarize my work on project ( New York City Taxi Trip Duration ) : kaggle competition
 
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
   - MY Score: ( 0.47672 ) ---> metric for this competition is Root Mean Squared Logarithmic Error ( RMSE ).

## Final: i made ( Map_App ) --> this is Web contain Map (Preferably applied on :( NYC ) )
   - Must write as similar this command ( streamlit run d:/Code_ML/Project_TaxiTripDuration/Map_App.py ) on the terminal to give you the Web .
   - select ( Pickup ) and ( Dropoff ) on the Map , then click on ( Predictioin Duration ) --> then it will predict the Time Duration for the Distance .

 
<img width="1266" height="607" alt="image" src="https://github.com/user-attachments/assets/3a3196c6-ba46-446d-92f3-d6571470b5a7" />
