

from flask import *
import pickle
import sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app=Flask(__name__)


def make_pred(inps):
    trf=pickle.load(open('column_trans.pkl', 'rb'))
    pipe=pickle.load(open('salary_pred.pkl','rb')) 

    data = pd.DataFrame([inps], columns=['job_title', 'experience_level', 'employment_type', 'work_models', 'employee_residence', 'company_location', 'company_size', 'work_year'])
    salary=pipe.predict(data)[0]
    return salary


@app.route("/")
def home_fun():

    return render_template("main.html")


@app.route("/pred_link",methods=["POST"])
def check_fun():
    job_title=request.form["job_title"]
    experience_level=request.form["experience_level"]
    employment_type=request.form["employment_type"]
    work_models=request.form["work_models"]
    employee_residence=request.form["employee_residence"]
    company_location=request.form["company_location"]
    company_size=request.form["company_size"]
    work_year=request.form["work_year"]
    
    inps=[job_title, experience_level, employment_type, work_models, employee_residence, company_location, company_size, work_year]
    
    result=make_pred(inps)
    return render_template("display.html",salary=result)



if (__name__=="__main__"):
    app.run(debug=True)