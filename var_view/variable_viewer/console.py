# var_view/variable_viewer/console.py

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
import logging
import re

logger = logging.getLogger(__name__)


class ConsoleManager:
    def __init__(self, data_source, alias, refresh_callback):
        self.data_source = data_source
        self.alias = alias
        self.refresh_callback = refresh_callback
        self.console_window = None
        self.kernel_manager = None
        self.kernel_client = None
        self.console_widget = None
        self.failed_lines = []
        self.setup_console()

    def setup_console(self):
        try:
            self.kernel_manager = QtInProcessKernelManager()
            self.kernel_manager.start_kernel()
            self.kernel_manager.kernel.gui = "qt"

            self.kernel_client = self.kernel_manager.client()
            self.kernel_client.start_channels()

            console = RichJupyterWidget()
            console.kernel_manager = self.kernel_manager
            console.kernel_client = self.kernel_client
            self.console_widget = console

            self.console_window = QWidget()
            self.console_window.setWindowTitle("Console")
            layout = QVBoxLayout(self.console_window)
            layout.addWidget(console)
            clear_btn = QPushButton("Clear Failed")
            clear_btn.clicked.connect(self.clear_failed)
            layout.addWidget(clear_btn)
            self.console_window.resize(600, 960)
            self.console_window.show()

            # Inject data_source
            kernel = self.kernel_manager.kernel  # 'kernel' is the shell

            if not hasattr(kernel, 'shell'):
                logger.exception("Kernel does not have a 'shell' attribute.")
                return

            shell = kernel.shell

            if not hasattr(shell, 'events'):
                logger.exception("Kernel shell does not have an 'events' attribute.")
                return

            shell.push({self.alias: self.data_source})

            # Define the event handler
            def refresh_after_execute(result):
                """
                Event handler triggered after a cell is executed.

                Parameters:
                - result: An ExecutionResult object containing execution details.
                """
                try:
                    # Extract the executed cell's source code
                    cell = result.info.raw_cell.strip()
                    logger.debug("Executed command: %s", cell)

                    if not result.success:
                        self.failed_lines.append(result.execution_count)

                    # Check if the command starts with f"{alias}."
                    if cell.startswith(f"{self.alias}."):
                        # Extract the parameter being accessed or assigned
                        param_match = re.match(
                            rf"{re.escape(self.alias)}\.([A-Za-z_][A-Za-z0-9_]*)", cell)
                        if param_match:
                            param_name = param_match.group(1)
                            full_param_name = f"{param_name}"

                            # Check if the parameter already exists in the viewer
                            if self.refresh_callback.has_variable(full_param_name):
                                logger.debug("Parameter '%s' already exists. No "
                                             "refresh needed.", full_param_name)
                            else:
                                logger.info("Parameter '%s' does not exist. "
                                            "Refreshing view.", full_param_name)
                                self.refresh_callback()
                        else:
                            logger.debug("Could not parse parameter from command: %s",
                                         cell)
                    else:
                        logger.debug("Command does not start with '%s': %s",
                                     self.alias, cell)
                except Exception as err:
                    logger.exception("Error during conditional refresh: %s", err)

            # Register the event handler with post_run_cell
            try:
                shell.events.register('post_run_cell', refresh_after_execute)
                logger.info("Registered 'post_run_cell' event handler.")
            except AttributeError as e:
                logger.exception("Failed to register event handler: %s", e)

            logger.info("Console window opened and '%s' injected.", self.alias)
        except Exception as e:
            logger.exception("Failed to set up console: %s", e)

    def clear_failed(self):
        """Remove failed commands from history and clear the display."""
        try:
            shell = self.kernel_manager.kernel.shell
            hist = shell.history_manager
            session_number = hist.session_number
            for line in self.failed_lines:
                hist.db.execute(
                    "DELETE FROM history WHERE session=? AND line=?",
                    (session_number, line),
                )
                hist.db.execute(
                    "DELETE FROM output_history WHERE session=? AND line=?",
                    (session_number, line),
                )
            hist.db.commit()
            self.failed_lines.clear()
            if self.console_widget is not None:
                self.console_widget.clear()
        except Exception as err:
            logger.exception("Failed to clear failed history: %s", err)
