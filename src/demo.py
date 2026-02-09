from src.pipeline import generate_pathfinder_suggestions
import json

tests = [
    ("18-24", "Salary"),
    ("25-34", "Hourly + Overtime"),
    ("45-54", "Self-Employed")
]

for age, income in tests:
    print("=" * 80)
    result = generate_pathfinder_suggestions(age, income)
    print(json.dumps(result, indent=2))






