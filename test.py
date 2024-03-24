import time
import sys

def print_progress(outer, total_outer, inner, total_inner):
    sys.stdout.write('\x1b[2J\x1b[H') 
    if outer >= 0: 
        print("Total progress:")
        percent_outer = outer / total_outer  
        bar_length_outer = int(30 * percent_outer)
        bar_outer = '[' + '=' * bar_length_outer + ' ' * (30 - bar_length_outer) + ']'
        print(f"{bar_outer} {percent_outer * 100:.2f}% ({outer}/{total_outer})")
    
    if inner >= 0:  
        print("Inner progress:")
        percent_inner = (inner + 1) / total_inner
        bar_length_inner = int(30 * percent_inner)
        bar_inner = '[' + '=' * bar_length_inner + ' ' * (30 - bar_length_inner) + ']'
        print(f"{bar_inner} {percent_inner * 100:.2f}% ({inner+1}/{total_inner})")

total_outer = 10
total_inner = 100

for outer in range(total_outer):
    for inner in range(total_inner):
        print_progress(outer, total_outer, inner, total_inner)  
        time.sleep(0.01)  
    print_progress(outer + 1, total_outer, -1, total_inner)  

print(f"\nCalculation for {} Finished.")
