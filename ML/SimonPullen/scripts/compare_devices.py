#!/usr/bin/env python3
"""
Compare CPU vs GPU performance for backtesting
"""

import subprocess
import json
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.device_manager import get_device_manager


def run_backtest(device: str, instrument: str = 'EUR_USD', 
                 days: int = 365, stop_type: str = 'aggressive'):
    """Run backtest with specified device"""
    
    cmd = [
        'python', 'scripts/backtest_patterns.py',
        '--instrument', instrument,
        '--timeframe', '1h',
        '--days', str(days),
        '--balance', '10000',
        '--stop-type', stop_type,
        '--device', device,
        '--output', f'backtest_{device}.json'
    ]
    
    print(f"\n{'='*60}")
    print(f"Running backtest on {device.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return {
        'device': device,
        'elapsed': elapsed,
        'success': result.returncode == 0,
        'output_file': f'backtest_{device}.json'
    }


def main():
    # Check available devices
    dm = get_device_manager('auto')
    info = dm.get_device_info()
    
    print("\n" + "="*60)
    print("DEVICE COMPARISON TEST")
    print("="*60)
    print(f"Current device: {info['name']}")
    print(f"Type: {info['type']}")
    
    # Test devices
    results = []
    devices_to_test = ['cpu']
    
    # Add GPU if available
    if info['type'] == 'gpu':
        devices_to_test.append(info['device'])
    
    for device in devices_to_test:
        result = run_backtest(device)
        results.append(result)
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    for result in results:
        speedup = "N/A"
        if result['device'] != 'cpu' and results[0]['device'] == 'cpu':
            cpu_time = results[0]['elapsed']
            speedup = cpu_time / result['elapsed'] if result['elapsed'] > 0 else 0
            speedup_str = f"{speedup:.2f}x faster"
        else:
            speedup_str = "baseline"
        
        print(f"{result['device'].upper()}: {result['elapsed']:.2f}s ({speedup_str})")
    
    # Load and compare results
    print("\n" + "="*60)
    print("RESULT COMPARISON")
    print("="*60)
    
    for result in results:
        try:
            with open(result['output_file'], 'r') as f:
                data = json.load(f)
                trades = data['results']['total_trades']
                win_rate = data['results']['win_rate']
                print(f"{result['device'].upper()}: {trades} trades, {win_rate:.1f}% win rate")
        except:
            print(f"{result['device'].upper()}: Could not load results")


if __name__ == '__main__':
    main()
