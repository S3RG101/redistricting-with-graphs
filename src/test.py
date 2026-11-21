from pums_loader import load_pums_persons

people = load_pums_persons(state="NY")   # pandas DataFrame, one row per person

print(people.head(5))

print(people.columns.tolist())