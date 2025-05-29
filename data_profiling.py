from ydata_profiling import ProfileReport
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('cardiotocography_v2.csv', encoding='utf-8')
    df['Tendency'] = pd.to_numeric(df["Tendency"])
    profiling = ProfileReport(df, title='Cardiotocography data report')
    profiling.to_file("data-profiling.html")

    json_data = profiling.to_json()

    # As a file
    profiling.to_file("your_report.json")
