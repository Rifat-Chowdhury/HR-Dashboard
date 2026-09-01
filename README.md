# HR Dashboard

![HR Dashboard logo](Images/Logo.png)

An interactive Tableau dashboard for exploring workforce composition, hiring and termination trends, employee demographics, and compensation. It is built with a realistic **synthetic** HR dataset and is intended for portfolio, learning, and demonstration use.

## Dashboard

Open [HR Dashboard.twbx](HR%20Dashboard.twbx) in Tableau Desktop or Tableau Public to use the dashboard. The packaged workbook includes the data extracts and image assets it needs, so it can be opened without connecting to an external data source.

The workbook was last saved with Tableau `2025.3.3`. Tableau may prompt to upgrade the workbook when it is opened in a newer version.

### Summary view

`HR | Summary` provides an at-a-glance view of the organization:

- Headline metrics for total hired, active, and terminated employees.
- Hiring and termination trends by year.
- Workforce distribution by department, gender, education level, age group, state, city, and HQ versus branch location.
- A U.S. state map and location comparison, where employees in New York are classified as `HQ` and all other states as `Branch`.
- Education versus performance, gender versus education, and age versus salary analyses.

Use the available filters for gender, employment status, location, hire year, and education level to refine the analysis.

### Employee details view

`HR | Details` is a filterable employee-level table. It includes employee ID, name, demographic attributes, role and department, location, salary, employment status, hire and termination dates, and length of employment.

The view provides filters for employee ID, name, gender, age group, education, job title, department, location, state, city, salary, status, hire date, termination date, and length of employment.

## Dataset

[HumanResources.csv](HumanResources.csv) contains 8,950 synthetic employee records spanning 2015-2024. The current data covers 11 departments, 53 job titles, 10 U.S. states, and 50 cities. It contains no real employee data.

| Field | Description |
| --- | --- |
| `Employee ID` | Unique identifier in the `##-########` format. |
| `First Name`, `Last Name` | Synthetic employee name. |
| `Gender` | Employee gender category. |
| `State`, `City` | U.S. employee location. |
| `Hire Date` | Employee start date. |
| `Department`, `Job Title` | Organizational role information. |
| `Education Level` | Highest education category. |
| `Performance Rating` | Performance category. |
| `Overtime` | Whether overtime is worked. |
| `Salary` | Base annual salary in USD. |
| `Birth Date` | Synthetic birth date used for age analysis. |
| `Termination Date` | Exit date; blank for active employees. |
| `Adjusted Salary` | Salary adjusted by the generator's gender, education, and age rules. |

The workbook derives the following fields:

- `Status`: `Hired` when `Termination Date` is blank; otherwise `Terminated`.
- `Total Hired`, `Total Active`, and `Total Terminated`: employee counts based on ID and termination status.
- `Location`: `HQ` for New York and `Branch` for every other state.
- `Age`, `Age Groups`, `Length of Hire`, and `Full Name` for analysis and display.

## Regenerate the data

The dataset generator has no third-party Python dependencies. It uses only the Python standard library.

```powershell
python DataGenerator.py
```

This command overwrites `HumanResources.csv` with a deterministic 8,950-record dataset because the script sets `SEED = 42`. Adjust the `CONFIG` section in [DataGenerator.py](DataGenerator.py) to change record counts, distributions, salary ranges, or the output file name. Set `SEED = None` to generate a different dataset on each run.

> Regenerating the CSV does not automatically update the packaged workbook. Open the `.twbx` in Tableau, replace or reconnect the data source to the updated CSV, refresh the extract, and save the workbook to include the new data.

## Project structure

```text
.
|- HR Dashboard.twbx       # Packaged Tableau workbook and embedded extracts
|- HumanResources.csv      # Synthetic HR source data
|- DataGenerator.py        # Standard-library CSV generator
|- ProjectRequirements.txt # Original dashboard user story and requirements
|- Images/Logo.png         # Project logo
`- LICENSE                 # MIT License
```

## Requirements

- Tableau Desktop or Tableau Public to open and interact with the workbook.
- Python 3.10+ to run the data generator. The type-hint syntax in the script requires Python 3.10 or later.

## License

This project is licensed under the [MIT License](LICENSE).
