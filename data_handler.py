import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataHandler:
    def __init__(self, data_file='tasks.csv'):
        self.data_file = data_file
        self.df = None

    def load_data(self):
        """Load historical task data from CSV"""
        if os.path.exists(self.data_file):
            self.df = pd.read_csv(self.data_file)
            self._preprocess_data()
            return self.df
        else:
            print(f"No data file found at {self.data_file}. Creating sample data.")
            return self._create_sample_data()

    def _preprocess_data(self):
        """Preprocess data: extract hour, day, calculate durations"""
        if self.df is not None:
            self.df['completion_date'] = pd.to_datetime(self.df['completion_date'])
            self.df['start_hour'] = self.df['start_time'].apply(lambda x: int(x.split(':')[0]))
            self.df['duration'] = self.df['actual_time'].apply(self._time_to_hours) - self.df['estimated_time'].apply(self._time_to_hours)
            self.df['day_of_week'] = self.df['completion_date'].dt.dayofweek

    def _time_to_hours(self, time_str):
        h, m = map(int, time_str.split(':'))
        return h + m/60

    def _create_sample_data(self):
        sample_data = {
            'task_name': ['Email responses', 'Code review', 'Meeting prep', 'Report writing', 'Bug fix', 'Planning'],
            'priority': [2, 1, 3, 1, 2, 3],
            'estimated_time': ['09:00', '10:30', '14:00', '11:00', '15:30', '08:00'],
            'actual_time': ['09:45', '11:20', '14:50', '12:30', '16:15', '08:40'],
            'completion_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20'],
            'start_time': ['09:00', '10:30', '14:00', '11:00', '15:30', '08:00']
        }
        self.df = pd.DataFrame(sample_data)
        self.df.to_csv(self.data_file, index=False)
        self._preprocess_data()
        return self.df

    def save_data(self, new_data):
        """Append new task data"""
        new_df = pd.DataFrame([new_data])
        if self.df is not None:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
        else:
            self.df = new_df
        self.df.to_csv(self.data_file, index=False)

    def get_user_patterns(self):
        """Extract behavior patterns"""
        if self.df is None:
            return {}
        patterns = {
            'avg_duration_by_hour': self.df.groupby('start_hour')['duration'].mean().to_dict(),
            'productivity_by_day': self.df.groupby('day_of_week')['priority'].mean().to_dict(),
            'preferred_hours': self.df['start_hour'].mode().tolist(),
            'task_freq_by_priority': self.df['priority'].value_counts().to_dict()
        }
        return patterns

