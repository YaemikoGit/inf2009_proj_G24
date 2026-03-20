import psutil
import time
import json
from datetime import datetime

def monitor_publisher():
    """Monitor the publisher process"""
    
    # Find the publisher process
    publisher_proc = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'])
            if 'mqtt_publisher' in cmdline:
                publisher_proc = psutil.Process(proc.info['pid'])
                break
        except:
            continue
    
    if not publisher_proc:
        print("Publisher process not found. Start mqtt_publisher.py first!")
        return
    
    print(f"Monitoring PID: {publisher_proc.pid}")
    print(f"Started at: {datetime.now()}")
    print("\nTime\t\tCPU%\tRAM(MB)\tThreads")
    print("-" * 60)
    
    log_data = []
    
    try:
        while True:
            # Get metrics
            cpu_percent = publisher_proc.cpu_percent(interval=1)
            mem_info = publisher_proc.memory_info()
            ram_mb = mem_info.rss / 1024 / 1024
            num_threads = publisher_proc.num_threads()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"{timestamp}\t{cpu_percent:.1f}\t{ram_mb:.1f}\t{num_threads}")
            
            # Log data
            log_data.append({
                "timestamp": timestamp,
                "cpu_percent": cpu_percent,
                "ram_mb": ram_mb,
                "threads": num_threads
            })
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        
        # Save log
        with open('resource_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Calculate statistics
        if log_data:
            cpu_values = [d['cpu_percent'] for d in log_data]
            ram_values = [d['ram_mb'] for d in log_data]
            
            print("\n=== RESOURCE USAGE SUMMARY ===")
            print(f"Duration: {len(log_data)} seconds")
            print(f"\nCPU Usage:")
            print(f"  Average: {sum(cpu_values)/len(cpu_values):.1f}%")
            print(f"  Peak: {max(cpu_values):.1f}%")
            print(f"\nRAM Usage:")
            print(f"  Average: {sum(ram_values)/len(ram_values):.1f} MB")
            print(f"  Peak: {max(ram_values):.1f} MB")
            print(f"\nLog saved to: resource_log.json")

if __name__ == "__main__":
    monitor_publisher()