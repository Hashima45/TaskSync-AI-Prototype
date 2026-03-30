# TaskSync AI

## Overview
TaskSync AI is an intelligent task scheduling system that uses machine learning to optimize your schedule based on historical task data and user behavior patterns.

## Features
- Analyzes task completion times, priorities, frequency
- Learns user productivity trends, preferred hours, habits
- Dynamically generates and adjusts schedules
- Improves efficiency and task prioritization

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Add your data to `tasks.csv` with columns: task_name, priority, estimated_time, actual_time, completion_date, start_time
5. Run: `python main.py`

## Usage
The system loads historical data, trains an ML model, generates optimized schedules based on predicted durations, user patterns (preferred hours, productivity by day/hour), and priorities. It includes sample data to start.

Update tasks.csv with your real data for better training.

