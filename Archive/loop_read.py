import os

results_dir = '/home/basel/Work/python/ai_text_detection/results'
read_count = 0

for i in range(5):
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'rb') as f:
                    f.read()
                read_count += 1
            except Exception as e:
                pass

print(f"Finished looping 5 times. Total file reads: {read_count}")
