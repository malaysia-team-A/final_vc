import psutil
import os

print("Killing python processes...")
killed = 0
for proc in psutil.process_iter(['pid', 'name']):
    try:
        if proc.info['name'] == 'python.exe':
            print(f"Killing {proc.info['pid']}")
            proc.kill()
            killed += 1
    except Exception as e:
        print(f"Error killing {proc.info.get('pid')}: {e}")

print(f"Total killed: {killed}. Re-launching server...")
