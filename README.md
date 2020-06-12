# Background

Assessments of biological (rather than chronological) age derived from patient biochemical data have been shown to strongly predict both all-cause and disease-specific mortality. However, these population-based approaches have yet to be translated to the individual. As well as using biological age as a research tool, by being able to better answer the question “why did we get this result?”, clinicians may be able to apply personalised interventions that could improve the long-term health of individual patients.

# Methodology and Data 

The boosted decision tree algorithm XGBoost was used to predict biological age using 39 commonly-available blood test results from the US National Health and Nutrition Examination Survey (NHANES) database.

# Results

Interrogation of the algorithm produced a description of how each marker contributed to the final output in a single individual. Additive explanation plots were then used to determine biomarker ranges associated with a lower biological age. Importantly, a number of markers that are modifiable with lifestyle changes were found to have a significant effect on biological age, including fasting blood glucose, lipids, and markers of red blood cell production.

# Conclusions

The combination of individualised outputs with target ranges could provide the ability to personalise interventions or recommendations based on an individual’s biochemistry and resulting predicted age. This would allow for the investigation of interventions designed to improve health and longevity in a targeted manner, many of which could be rooted in targeted lifestyle modifications.

# Install

```
git clone https://github.com/cck197/ml-bio-age.git
virtualenv venv -p python3
. venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

# Run

Point your web browser at the Jupyter notebook server running locally, e.g. `http://127.0.0.1:8888` then run the notebooks in the directory `nbs/` in the order listed.
