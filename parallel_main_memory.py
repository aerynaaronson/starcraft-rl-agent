"""
Parallel training with memory-enabled LSTM.

This is a drop-in replacement for parallel_main.py that uses
the memory-enabled architecture for better long-term learning.
"""

import torch.multiprocessing as mp
from gpu_server_memory import gpu_server_process
from worker_memory import game_worker
import time
import psutil

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    # Configuration
    NUM_WORKERS = 8
    EPISODES_PER_WORKER = 1250
    
    # Setup stdout buffering
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    print("="*60)
    print("PARALLEL TRAINING - MEMORY-ENABLED LSTM")
    print("="*60)
    print(f"Workers: {NUM_WORKERS}")
    print(f"Episodes per worker: {EPISODES_PER_WORKER}")
    print(f"Total episodes: {NUM_WORKERS * EPISODES_PER_WORKER}")
    print()
    print("Architecture:")
    print("  - 3-layer LSTM (768 hidden units)")
    print("  - Temporal attention (8 heads)")
    print("  - Episode memory buffer (2000 steps)")
    print("  - Sequence training (128-step sequences)")
    print("  - Estimated VRAM: ~4-6GB of 24GB")
    print("="*60)
    
    mem = psutil.virtual_memory()
    print(f"Initial RAM: {mem.percent:.1f}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")
    print()
    
    # Create queues
    prediction_queue = mp.Queue(maxsize=200)
    result_queue = mp.Queue(maxsize=200)
    update_queue = mp.Queue(maxsize=100)
    control_queue = mp.Queue()
    
    # Start GPU server
    print("Starting GPU Server (memory-enabled)...")
    gpu_server = mp.Process(
        target=gpu_server_process,
        args=(prediction_queue, result_queue, update_queue, control_queue),
        daemon=False
    )
    gpu_server.start()
    time.sleep(5)  # Give GPU server time to initialize
    print("✓ GPU Server running\n")
    
    # Start workers
    print(f"Starting {NUM_WORKERS} game workers...")
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=game_worker,
            args=(i, prediction_queue, result_queue, update_queue, EPISODES_PER_WORKER),
            daemon=False
        )
        p.start()
        workers.append(p)
        print(f"  Worker {i} started")
        time.sleep(3)  # Stagger worker starts
    
    print()
    print("="*60)
    print("ALL WORKERS LAUNCHED")
    print("Training in progress... Press Ctrl+C to stop gracefully")
    print("="*60)
    print()
    
    start_time = time.time()
    
    try:
        # Wait for all workers to complete
        for i, w in enumerate(workers):
            w.join()
            elapsed = time.time() - start_time
            print(f"Worker {i} completed ({elapsed/60:.1f} minutes elapsed)")
        
        print(f"\n✓ All workers finished in {(time.time() - start_time)/60:.1f} minutes!")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("KEYBOARD INTERRUPT - SAVING MODEL...")
        print("="*60)
        
        print("\n1. Sending SAVE command to GPU server...")
        try:
            control_queue.put('SAVE:memory_model_interrupted.pth', timeout=1)
        except:
            print("   Warning: Could not send save command")
        time.sleep(3)
        print("   Model save command sent")
        
        print("\n2. Terminating workers...")
        for i, w in enumerate(workers):
            if w.is_alive():
                print(f"   Terminating worker {i}")
                w.terminate()
        time.sleep(2)
    
    finally:
        print("\n" + "="*60)
        print("CLEANUP PHASE")
        print("="*60)
        
        print("\n3. Stopping GPU server...")
        try:
            control_queue.put('STOP', timeout=1)
        except:
            pass
        
        if gpu_server.is_alive():
            print("   Waiting for GPU server to finish...")
            gpu_server.join(timeout=10)
        
        if gpu_server.is_alive():
            print("   GPU server not responding, forcing termination...")
            gpu_server.terminate()
            gpu_server.join(timeout=3)
        else:
            print("   ✓ GPU server stopped cleanly")
        
        print("\n4. Killing SC2 processes...")
        import subprocess
        try:
            if psutil.WINDOWS:
                result = subprocess.run(['taskkill', '/F', '/IM', 'SC2_x64.exe'], 
                             capture_output=True, timeout=5)
                subprocess.run(['taskkill', '/F', '/IM', 'SC2.exe'], 
                             capture_output=True, timeout=5)
            else:
                result = subprocess.run(['pkill', '-9', 'SC2'], 
                             capture_output=True, timeout=5)
            print(f"   ✓ SC2 processes cleaned up")
        except Exception as e:
            print(f"   Warning: {e}")
        
        print("\n5. Checking for saved models...")
        import os
        models = []
        for f in os.listdir('.'):
            if f.endswith('.pth'):
                size = os.path.getsize(f) / 1024 / 1024
                models.append((f, size))
        
        if models:
            print("   ✓ Found saved models:")
            for name, size in sorted(models):
                print(f"     - {name} ({size:.1f} MB)")
        else:
            print("   ⚠ No .pth files found - model may not have saved!")
        
        mem = psutil.virtual_memory()
        print(f"\nFinal RAM: {mem.percent:.1f}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")
        print()
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Check memory_model_final.pth or latest checkpoint")
        print("2. Compare performance with your old model")
        print("3. Look for faster learning and fewer steps to victory!")