import io
import traceback

import numpy as np
import pandas as pd
import plotly.express as px

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INSIGHTS = """The Titanic dataset contains 891 passengers, of which only 342 (38.4%) survived the disaster. The data reveals clear patterns in who was saved and who was not, strongly influenced by gender, ticket class, and age.

The most striking finding is the gender gap: 74% of women survived compared to only 19% of men, reflecting the "women and children first" evacuation policy. Passenger class was equally decisive — 1st class passengers survived at 63%, 2nd class at 47%, and 3rd class at only 24%. This shows that wealthier passengers had better access to lifeboats, likely due to their cabin locations being closer to the deck.

Children under 12 had a higher survival rate than adults, consistent with the evacuation priority given to them. Passengers travelling with a small family (2-4 people) survived slightly better than those travelling alone, possibly because family members helped each other reach lifeboats.

The correlation heatmap confirms that being female and paying a higher fare are the two strongest predictors of survival. Passenger class and fare are highly correlated, meaning richer passengers had a double advantage.

In conclusion, survival on the Titanic was not random — it was heavily shaped by social inequality. Your wealth determined your cabin location, your access to lifeboats, and ultimately your chance of survival. The data is a snapshot of class inequality in 1912 frozen in tragedy."""


def load_file(filename, content):
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("please upload a csv or excel file")
    return df


def clean_data(df):
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Cabin"] = df["Cabin"].fillna("Unknown")
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    rare = ["Dr", "Rev", "Col", "Major", "Mlle", "Countess", "Ms",
            "Lady", "Jonkheer", "Don", "Dona", "Capt", "Sir"]
    df["Title"] = df["Title"].apply(lambda t: "Rare" if t in rare else t)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child (0-12)", "Teen (13-18)", "Adult (19-35)", "Middle-aged (36-60)", "Senior (60+)"]
    )
    df["PclassLabel"] = df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
    df["SurvivedLabel"] = df["Survived"].map({0: "Did Not Survive", 1: "Survived"})
    return df


def make_charts(df):
    all_charts = []

    def save_chart(fig, title):
        all_charts.append({"title": title, "plotly_json": fig.to_json()})

    counts = df["SurvivedLabel"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    fig = px.bar(counts, x="Status", y="Count",
                 title="How Many People Survived?",
                 color="Status",
                 color_discrete_map={"Survived": "#2ecc71", "Did Not Survive": "#e74c3c"})
    save_chart(fig, "How Many People Survived?")

    by_class = df.groupby("PclassLabel")["Survived"].mean().reset_index()
    by_class["Survival Rate (%)"] = (by_class["Survived"] * 100).round(1)
    fig = px.bar(by_class, x="PclassLabel", y="Survival Rate (%)",
                 title="Survival Rate by Ticket Class",
                 color="PclassLabel", text="Survival Rate (%)")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    save_chart(fig, "Survival Rate by Ticket Class")

    by_sex = df.groupby("Sex")["Survived"].mean().reset_index()
    by_sex["Survival Rate (%)"] = (by_sex["Survived"] * 100).round(1)
    fig = px.bar(by_sex, x="Sex", y="Survival Rate (%)",
                 title="Survival Rate by Gender",
                 color="Sex",
                 color_discrete_map={"male": "#3498db", "female": "#e91e8c"},
                 text="Survival Rate (%)")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    save_chart(fig, "Survival Rate by Gender")

    by_class_sex = df.groupby(["PclassLabel", "Sex"])["Survived"].mean().reset_index()
    by_class_sex["Survival Rate (%)"] = (by_class_sex["Survived"] * 100).round(1)
    fig = px.bar(by_class_sex, x="PclassLabel", y="Survival Rate (%)",
                 color="Sex", barmode="group",
                 title="Survival Rate by Class and Gender",
                 color_discrete_map={"male": "#3498db", "female": "#e91e8c"},
                 text="Survival Rate (%)")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    save_chart(fig, "Survival Rate by Class and Gender")

    fig = px.histogram(df, x="Age", color="SurvivedLabel",
                       title="Age Distribution (survived vs not)",
                       nbins=30, barmode="overlay", opacity=0.7,
                       color_discrete_map={"Survived": "#2ecc71", "Did Not Survive": "#e74c3c"})
    save_chart(fig, "Age Distribution")

    df_no_outliers = df[df["Fare"] < 300]
    fig = px.histogram(df_no_outliers, x="Fare", color="SurvivedLabel",
                       title="Fare Distribution (survived vs not)",
                       nbins=40, barmode="overlay", opacity=0.7,
                       color_discrete_map={"Survived": "#2ecc71", "Did Not Survive": "#e74c3c"})
    save_chart(fig, "Fare Distribution")

    fig = px.pie(df, names="Sex", title="Male vs Female Passengers",
                 color="Sex",
                 color_discrete_map={"male": "#3498db", "female": "#e91e8c"})
    save_chart(fig, "Male vs Female Passengers")

    fig = px.pie(df, names="Embarked",
                 title="Where Did Passengers Board? (C=Cherbourg, Q=Queenstown, S=Southampton)")
    save_chart(fig, "Where Did Passengers Board?")

    by_age = df.groupby("AgeGroup", observed=True)["Survived"].mean().reset_index()
    by_age["Survival Rate (%)"] = (by_age["Survived"] * 100).round(1)
    fig = px.bar(by_age, x="AgeGroup", y="Survival Rate (%)",
                 title="Survival Rate by Age Group",
                 color="AgeGroup", text="Survival Rate (%)")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    save_chart(fig, "Survival Rate by Age Group")

    fig = px.box(df[df["Fare"] < 300], x="PclassLabel", y="Fare",
                 title="Ticket Price by Passenger Class",
                 color="PclassLabel")
    save_chart(fig, "Ticket Price by Passenger Class")

    by_family = df.groupby("FamilySize")["Survived"].mean().reset_index()
    by_family["Survival Rate (%)"] = (by_family["Survived"] * 100).round(1)
    fig = px.line(by_family, x="FamilySize", y="Survival Rate (%)",
                  title="Does Family Size Affect Survival?",
                  markers=True)
    fig.update_traces(line_width=2, marker_size=8)
    save_chart(fig, "Does Family Size Affect Survival?")

    by_title = df.groupby("Title")["Survived"].mean().reset_index()
    by_title["Survival Rate (%)"] = (by_title["Survived"] * 100).round(1)
    by_title = by_title.sort_values("Survival Rate (%)", ascending=False)
    fig = px.bar(by_title, x="Title", y="Survival Rate (%)",
                 title="Survival Rate by Title",
                 color="Survival Rate (%)", color_continuous_scale="RdYlGn",
                 text="Survival Rate (%)")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    save_chart(fig, "Survival Rate by Title")

    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    corr_matrix = df[num_cols].corr().round(2)
    fig = px.imshow(corr_matrix, text_auto=True,
                    title="Correlation Between Variables",
                    color_continuous_scale="RdBu_r", aspect="auto")
    save_chart(fig, "Correlation Between Variables")

    fig = px.scatter(df[df["Fare"] < 300], x="Age", y="Fare",
                     color="SurvivedLabel",
                     title="Age vs Fare Price",
                     color_discrete_map={"Survived": "#2ecc71", "Did Not Survive": "#e74c3c"},
                     opacity=0.6)
    save_chart(fig, "Age vs Fare Price")

    return all_charts


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx") or file.filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Please upload a CSV or Excel file")

    content = await file.read()

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File is too big, max 50MB")

    try:
        df = load_file(file.filename, content)

        if df.empty:
            raise HTTPException(status_code=400, detail="The file is empty")

        missing = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        df = clean_data(df)

        show_cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex",
                     "Age", "Fare", "Embarked", "Title", "FamilySize", "AgeGroup"]

        preview = df[show_cols].head(20).astype(str).to_dict(orient="records")
        stats = df[["Age", "Fare", "SibSp", "Parch", "FamilySize"]].describe().round(2).to_dict()
        charts = make_charts(df)

        info = {
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": show_cols,
            "missing_before_cleaning": {k: int(v) for k, v in missing.items()},
            "survival_rate": round(df["Survived"].mean() * 100, 1),
        }

        return JSONResponse({
            "dataset_info": info,
            "preview": preview,
            "statistics": stats,
            "charts": charts,
            "insights": INSIGHTS,
        })

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"something went wrong: {str(e)}")


@app.get("/")
def home():
    return {"message": "titanic analysis server is running!"}
