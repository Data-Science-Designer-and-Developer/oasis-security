
import xgboost as xgb
from sklearn.model_selection import train_test_split
from preprocess import load_clean
from features import make_features
import joblib

df = load_clean("/home/frederic/Documents/Jedha/Jedha/02_Data Science and Eng - Fullstack - Full-Time/X_Projects/Project_Oasis/oasis-security-complete/data/crimes-et-délits-PN-GN.xlsx")
df = make_features(df)

X = df[["annee","ville_enc","nb"]]
y = df["crime_enc"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss")
model.fit(X_train,y_train)

joblib.dump(model,"model.joblib")
