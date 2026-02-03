import pandas as pd
import numpy as np

class MortalityDataHandler:
    def __init__(self, filepath):
        """
        Expects HMD 'Deaths' or 'Exposures' format.
        Columns: Year, Age, Female, Male, Total
        """
        self.raw_data = pd.read_csv(filepath, sep='\s+', skiprows=2)
        self.matrix = None

    def preprocess(self, gender='Total', min_age=0, max_age=90):
        df = self.raw_data.copy()

        # 1. Clean Age column
        df['Age'] = pd.to_numeric(
            df['Age'].astype(str).str.replace('+', '', regex=False),
            errors='coerce'
        )

        # 2. Ensure Year is numeric (important)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        # 3. Filter age range
        df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]

        # 4. Pivot into Age-Period matrix
        self.matrix = df.pivot(index='Age', columns='Year', values=gender)

        # 5. Force numeric dtype
        self.matrix = self.matrix.apply(pd.to_numeric, errors='coerce')

        # 6. Handle missing values
        self.matrix = self.matrix.replace(0, np.nan).interpolate(axis=1)

        return self.matrix


    def get_log_mortality(self):
        self.matrix = self.matrix.clip(lower=1e-10)
        return np.log(self.matrix)
