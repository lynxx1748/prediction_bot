#!/usr/bin/env python3
"""
Simple launcher for the trading bot (headless mode).
"""

import sys
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Bot Launcher')
    parser.add_argument('--mode', choices=['live', 'test', 'ask'], default='ask',
                       help='Trading mode: live (real money), test (simulation), or ask (prompt at startup)')
    return parser.parse_args()

def confirm_live_mode():
    """Get explicit confirmation for live mode."""
    print("\n⚠️ WARNING: LIVE MODE WILL PLACE REAL BETS WITH REAL MONEY! ⚠️\n")
    print("Type 'I CONFIRM' (all caps) to proceed with live mode, or anything else to switch to test mode:")
    
    confirmation = input("> ")
    return confirmation == "I CONFIRM"

def main():
    """Main launcher function."""
    args = parse_arguments()
    
    # Determine trading mode
    mode = args.mode
    if mode == 'ask':
        print("\n🔄 Select trading mode:")
        print("1. Test mode (simulated bets, no real money)")
        print("2. Live mode (REAL bets with REAL money)")
        choice = input("\nSelect mode (1/2): ").strip()
        
        if choice == '2':
            # Get confirmation for live mode
            if confirm_live_mode():
                mode = 'live'
                print("✅ Live mode confirmed")
            else:
                mode = 'test'
                print("⚠️ Live mode not confirmed. Defaulting to test mode.")
        else:
            mode = 'test'
            print("✅ Test mode selected")
    
    # Start bot directly in headless mode
    print(f"🔄 Starting bot in headless mode ({mode})...")
    from main import main as run_main
    sys.exit(run_main(mode=mode))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting by user request...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 