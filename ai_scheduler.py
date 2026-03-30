import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
from data_handler import DataHandler

class AIScheduler:
    def __init__(self):
        self.data_handler = DataHandler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_file = 'scheduler_model.joblib'
        self.is_trained = False

    def train_model(self):
        """Train AI model on historical data"""
        df = self.data_handler.load_data()
        if len(df) < 3:
            print("Insufficient data for training. Using default predictions.")
            return False

        # Features: priority, start_hour, day_of_week
        # Target: duration
        X = df[['priority', 'start_hour', 'day_of_week']]
        y = df['duration']

        if len(X) > 3:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
            print(f"Model R^2 score: {score:.2f}")
        else:
            # Train on all data if small dataset
            self.model.fit(X, y)
            print("Trained on full small dataset.")

        dump(self.model, self.model_file)
        self.is_trained = True
        return True

    def load_model(self):
        """Load pre-trained model"""
        try:
            self.model = load(self.model_file)
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

    def predict_duration(self, priority, start_hour, day_of_week):
        """Predict task duration in hours"""
        if not self.is_trained:
            self.load_model() or self.train_model()
        features = np.array([[priority, start_hour, day_of_week]])
        pred = self.model.predict(features)[0]
        return max(pred, 0.5)  # Min 30 min

    def generate_schedule(self, tasks, day_of_week=0):
        """Generate optimized schedule for given tasks"""
        patterns = self.data_handler.get_user_patterns()
        schedule_list = []
        current_hour = 9.0  # Start at 9 AM
        max_end = 18.0  # End by 6 PM

        # Sort by priority descending
        tasks_sorted = sorted(tasks, key=lambda x: x.get('priority', 3), reverse=True)

        for task in tasks_sorted:
            if current_hour >= max_end:
                break
            pred_duration = self.predict_duration(task['priority'], int(current_hour), day_of_week)
            end_hour = min(current_hour + pred_duration, max_end)
            
            # Adjust to preferred hour if possible
            preferred = patterns.get('preferred_hours')
            if preferred and current_hour not in preferred:
                current_hour = min(preferred[0], max_end - pred_duration)

            schedule_list.append({
                'name': task['name'],
                'priority': task['priority'],
                'start_hour': current_hour,
                'end_hour': end_hour,
                'predicted_duration': round(pred_duration, 1)
            })
            current_hour = end_hour + 0.25  # 15 min break

        return schedule_list

    def print_schedule(self, schedule):
        """Print schedule nicely"""
        print("\n=== Optimized Daily Schedule ===")
        for item in schedule:
            start = f"{int(item['start_hour']):02d}:{int((item['start_hour']%1)*60):02d}"
            end = f"{int(item['end_hour']):02d}:{int((item['end_hour']%1)*60):02d}"
            print(f"{start} - {end}: {item['name']} (P{item['priority']}, ~{item['predicted_duration']}h)")
        print("===============================\n")

