# Folder structure

.
├── Dockerfile
├── Makefile
├── README.md
├── api_gateway.yml                       # The configuration used in the API gateway in GCP
├── artifacts
│   └── model.pkl                         # The serialized model
├── challenge
│   ├── __init__.py
│   ├── api.py                            # The API with the routes
│   ├── constants.py                      # All the constants used in the model steps
│   ├── data_validation.py                # Stores a function to validate if the data is consistent
│   ├── exploration.ipynb
│   ├── model.py                          # A couple of methods were added: load_model and save_model
│   └── preprocessing.py                  # Keeps functions to generate features and the target columns
├── data
│   └── data.csv
├── docs
│   └── challenge.md
├── reports
├── requirements-dev.txt
├── requirements-test.txt
├── requirements.txt
├── setup.py
├── tests 
├── .github
│   └── workflow
│        ├── cd.yml                        # CD pipeline with manual trigger to deploy to cloud run
│        └── ci.yml                        # CI pipeline
└── workflows
    ├── cd.yml
    └── ci.yml

# API 
- The API was developed using Cloud Run and Api gateway

# Improvements
- I would like to develop a training pipeline using airflow or kubeflow
- All the GCP components like services accounts, cloud run and api gateway should have been created using TF, but I didn't have enought time
- Store all the artifacts in GCS or another cloud