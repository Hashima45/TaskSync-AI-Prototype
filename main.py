#!/usr/bin/env python3
"""
TaskSync AI Main Entry Point
"""
from data_handler import DataHandler
from ai_scheduler import AIScheduler
import schedule
import time
import pandas as pd

def main():
    print("🚀 TaskSync AI - AI-Powered Task Scheduler")
    print("Loading data and training model...\n")

    # Initialize
    scheduler = AIScheduler()
    trained = scheduler.train_model()

    if trained:
        print("✅ AI Model trained successfully!")
    else:
        print("⚠️  Using default predictions (add more data to tasks.csv for better results)")

    # Sample today's tasks (user can modify)
    today_tasks = [
        {'name': 'Daily standup meeting', 'priority': 3},
        {'name': 'Process emails & admin', 'priority': 2},
        {'name': 'Develop new feature', 'priority': 1},
        {'name': 'Code review PRs', 'priority': 1},
        {'name': 'Weekly report', 'priority': 2},
    ]

    # Generate and print schedule
    sched = scheduler.generate_schedule(today_tasks)
    scheduler.print_schedule(sched)

    # Interactive: Add new task example
    print("Example: Adding a new completed task updates the model next run.")
    # scheduler.data_handler.save_data({
    #     'task_name': 'Test task',
    #     'priority': 2,
    #     'estimated_time': '13:00',
    #     'actual_time': '13:45',
    #     'completion_date': '2024-10-04',
    #     'start_time': '13:00'
    # })

    print("💡 Tip: Edit tasks.csv with your real data and re-run to retrain.")
    print("Press Ctrl+C to exit.")

    # Optional: Schedule daily run at 8 AM
    # schedule.every().day.at("08:00").do(lambda: print("Daily schedule generated!"))
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)

if __name__ == "__main__":
    main()

