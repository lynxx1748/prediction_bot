"""
Desktop GUI for the trading bot.
Provides a Tkinter-based control panel with a modern interface.
"""

import json
import logging
import tkinter as tk
import traceback
from threading import Thread
from tkinter import ttk

from PIL import Image, ImageTk

from ..core import config
from ..data.database import get_recent_trades
from ..wallet import get_wallet_balance

logger = logging.getLogger(__name__)


class UglyBotControlPanel:
    """Main control panel for the UglyBot trading system."""

    def __init__(self, root):
        """
        Initialize the control panel.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("UglyBot Control Center")
        self.root.configure(bg="#0D1117")

        # Configure the style
        self.setup_styles()

        # Main container
        self.main_frame = ttk.Frame(root, style="Holo.TFrame")
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Create UI components
        self.status_frame = self.create_status_display()
        self.controls_frame = self.create_control_panels()
        self.metrics_frame = self.create_performance_metrics()
        self.log_frame = self.create_log_viewer()

        # Start data refresh
        self.refresh_data()

    def setup_styles(self):
        """Configure custom styles for the UI."""
        style = ttk.Style()
        style.configure(
            "Holo.TFrame", background="#1B2838", borderwidth=2, relief="solid"
        )
        style.configure(
            "Holo.TLabel",
            background="#1B2838",
            foreground="#66C0F4",
            font=("Arial", 10),
        )
        style.configure(
            "Title.TLabel",
            background="#1B2838",
            foreground="#66C0F4",
            font=("Arial", 14, "bold"),
        )

    def create_status_display(self):
        """Create the status display panel."""
        try:
            frame = ttk.Frame(self.main_frame, style="Holo.TFrame")
            frame.pack(padx=10, pady=10, fill="x")

            # Title
            title = ttk.Label(frame, text="System Status", style="Title.TLabel")
            title.pack(pady=5)

            # Status indicators
            self.status_bot = ttk.Label(frame, text="Bot: Offline", style="Holo.TLabel")
            self.status_bot.pack(anchor="w", padx=10, pady=2)

            self.status_wallet = ttk.Label(
                frame, text="Wallet: Not Connected", style="Holo.TLabel"
            )
            self.status_wallet.pack(anchor="w", padx=10, pady=2)

            self.status_network = ttk.Label(
                frame, text="Network: Unknown", style="Holo.TLabel"
            )
            self.status_network.pack(anchor="w", padx=10, pady=2)

            return frame
        except Exception as e:
            logger.error(f"Error creating status display: {e}")
            traceback.print_exc()
            return ttk.Frame(self.main_frame)

    def create_control_panels(self):
        """Create the control panels section."""
        try:
            frame = ttk.Frame(self.main_frame, style="Holo.TFrame")
            frame.pack(padx=10, pady=10, fill="x")

            # Title
            title = ttk.Label(frame, text="Controls", style="Title.TLabel")
            title.pack(pady=5)

            # Control buttons
            btn_frame = ttk.Frame(frame, style="Holo.TFrame")
            btn_frame.pack(padx=5, pady=5, fill="x")

            # Start button
            self.btn_start = tk.Button(
                btn_frame,
                text="Start Bot",
                bg="#1E5128",
                fg="white",
                padx=10,
                pady=5,
                command=self.start_bot,
            )
            self.btn_start.pack(side="left", padx=5, pady=5)

            # Stop button
            self.btn_stop = tk.Button(
                btn_frame,
                text="Stop Bot",
                bg="#8B0000",
                fg="white",
                padx=10,
                pady=5,
                command=self.stop_bot,
            )
            self.btn_stop.pack(side="left", padx=5, pady=5)

            # Settings button
            self.btn_settings = tk.Button(
                btn_frame,
                text="Settings",
                bg="#2C3E50",
                fg="white",
                padx=10,
                pady=5,
                command=self.open_settings,
            )
            self.btn_settings.pack(side="left", padx=5, pady=5)

            return frame
        except Exception as e:
            logger.error(f"Error creating control panels: {e}")
            traceback.print_exc()
            return ttk.Frame(self.main_frame)

    def create_performance_metrics(self):
        """Create the performance metrics section."""
        try:
            frame = ttk.Frame(self.main_frame, style="Holo.TFrame")
            frame.pack(padx=10, pady=10, fill="both", expand=True)

            # Title
            title = ttk.Label(frame, text="Performance Metrics", style="Title.TLabel")
            title.pack(pady=5)

            # Metrics display
            metrics_frame = ttk.Frame(frame, style="Holo.TFrame")
            metrics_frame.pack(padx=5, pady=5, fill="both", expand=True)

            # Win rate
            self.lbl_win_rate = ttk.Label(
                metrics_frame, text="Win Rate: N/A", style="Holo.TLabel"
            )
            self.lbl_win_rate.pack(anchor="w", padx=10, pady=2)

            # Total trades
            self.lbl_total_trades = ttk.Label(
                metrics_frame, text="Total Trades: 0", style="Holo.TLabel"
            )
            self.lbl_total_trades.pack(anchor="w", padx=10, pady=2)

            # Profit/Loss
            self.lbl_profit_loss = ttk.Label(
                metrics_frame, text="Profit/Loss: 0.0 BNB", style="Holo.TLabel"
            )
            self.lbl_profit_loss.pack(anchor="w", padx=10, pady=2)

            return frame
        except Exception as e:
            logger.error(f"Error creating performance metrics: {e}")
            traceback.print_exc()
            return ttk.Frame(self.main_frame)

    def create_log_viewer(self):
        """Create the log viewer section."""
        try:
            frame = ttk.Frame(self.main_frame, style="Holo.TFrame")
            frame.pack(padx=10, pady=10, fill="both", expand=True)

            # Title
            title = ttk.Label(frame, text="Log Viewer", style="Title.TLabel")
            title.pack(pady=5)

            # Log text area
            self.log_text = tk.Text(
                frame, bg="#0A1624", fg="#66C0F4", height=10, width=50
            )
            self.log_text.pack(padx=5, pady=5, fill="both", expand=True)

            # Add scrollbar
            scrollbar = ttk.Scrollbar(self.log_text)
            scrollbar.pack(side="right", fill="y")

            # Connect scrollbar to text widget
            self.log_text.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=self.log_text.yview)

            return frame
        except Exception as e:
            logger.error(f"Error creating log viewer: {e}")
            traceback.print_exc()
            return ttk.Frame(self.main_frame)

    def refresh_data(self):
        """Refresh all data in the UI."""
        try:
            # Update wallet status
            try:
                balance = get_wallet_balance()
                self.status_wallet.config(text=f"Wallet: {balance:.6f} BNB")
            except:
                self.status_wallet.config(text="Wallet: Error")

            # Update performance metrics
            try:
                # This would need to be replaced with actual data from your system
                trades = get_recent_trades()
                if trades:
                    wins = sum(1 for t in trades if t.get("won"))
                    total = len(trades)
                    win_rate = (wins / total) * 100 if total > 0 else 0
                    profit_loss = sum(float(t.get("profit_loss", 0)) for t in trades)

                    self.lbl_win_rate.config(text=f"Win Rate: {win_rate:.2f}%")
                    self.lbl_total_trades.config(text=f"Total Trades: {total}")
                    self.lbl_profit_loss.config(
                        text=f"Profit/Loss: {profit_loss:.6f} BNB"
                    )
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")

            # Schedule next refresh
            self.root.after(30000, self.refresh_data)  # Refresh every 30 seconds
        except Exception as e:
            logger.error(f"Error in refresh_data: {e}")
            traceback.print_exc()
            # Try to reschedule even after error
            self.root.after(30000, self.refresh_data)

    def start_bot(self):
        """Start the trading bot."""
        logger.info("Starting bot")
        # Bot starting logic would go here
        self.status_bot.config(text="Bot: Running")
        self.log_text.insert(
            tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Bot started\n"
        )
        self.log_text.see(tk.END)

    def stop_bot(self):
        """Stop the trading bot."""
        logger.info("Stopping bot")
        # Bot stopping logic would go here
        self.status_bot.config(text="Bot: Stopped")
        self.log_text.insert(
            tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Bot stopped\n"
        )
        self.log_text.see(tk.END)

    def open_settings(self):
        """Open the settings dialog."""
        logger.info("Opening settings")
        # Settings dialog would go here
        self.log_text.insert(
            tk.END,
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Opening settings\n",
        )
        self.log_text.see(tk.END)


def start_control_panel():
    """
    Start the desktop control panel in a separate thread.

    Returns:
        Thread: The control panel thread
    """
    try:

        def run_ui():
            root = tk.Tk()
            app = UglyBotControlPanel(root)
            root.geometry("800x600")
            root.mainloop()

        ui_thread = Thread(target=run_ui)
        ui_thread.daemon = True
        ui_thread.start()
        logger.info("Desktop control panel started")
        return ui_thread
    except Exception as e:
        logger.error(f"Failed to start desktop control panel: {e}")
        traceback.print_exc()
        return None
